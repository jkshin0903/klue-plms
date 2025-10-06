"""
klue/roberta-large 기반 KLUE-DP(의존 구문 분석) 파인튜닝 스크립트.

입력 데이터 형식 (간단/호환 가정):
- train.json, dev.json, test.json: 샘플 리스트
- 각 샘플 예시:
  {
    "tokens": ["나는", "사과를", "먹었다"],
    "heads": [2, 3, 0],  # 각 토큰의 head 인덱스(1-based, 0은 ROOT)
    "deprels": ["nsubj", "obj", "root"]
  }

간단한 모델 구성:
- Transformer(klue/roberta-large) 인코더 위에 의존/지배 쌍 점수를 위한 biaffine 헤드와
  관계 라벨 분류를 위한 biaffine 라벨러를 얹어 학습합니다.
  (구현 단순화를 위해 라벨 예측은 gold head를 사용해 학습, 추론 시 예측 head 사용)

실행 예시:
  python dp.py \
    --data_dir ./data/klue-dp \
    --output_dir ./outputs/dp \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from utils import configure_logging, get_model_name, save_label_list, set_seed, setup_cuda


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_klue_dp(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """KLUE-DP 데이터셋을 단순 포맷으로 로드합니다."""
    def pick(samples: List[Dict]) -> List[Dict]:
        out = []
        for s in samples:
            out.append(
                {
                    "tokens": s.get("tokens", []),
                    "heads": s.get("heads", []),  # 1-based head, 0 for ROOT
                    "deprels": s.get("deprels", []),
                }
            )
        return out

    train = pick(_read_json(os.path.join(data_dir, "train.json")))
    dev = pick(_read_json(os.path.join(data_dir, "dev.json")))
    test_path = os.path.join(data_dir, "test.json")
    test = pick(_read_json(test_path)) if os.path.exists(test_path) else []
    return train, dev, test


def build_deprel_list(datasets: Tuple[List[Dict], List[Dict], List[Dict]]) -> List[str]:
    labels = set(["root"])  # 기본값 포함
    for split in datasets:
        for s in split:
            labels.update(s.get("deprels", []))
    return sorted(labels)


def tokenize_and_align_for_dp(examples, tokenizer, deprel_to_id):
    # 단어 단위 입력을 워드피스에 매핑하고, 각 단어의 첫 서브워드 위치만 학습에 사용
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        return_offsets_mapping=False,
    )

    batch_heads = []  # 0-based head 인덱스, 0은 ROOT 가상 토큰으로 취급
    batch_deprels = []
    batch_word_starts = []  # 각 위치가 단어의 첫 서브워드인지 표시 (1/0)

    for i, (heads, deprels) in enumerate(zip(examples["heads"], examples["deprels"])):
        word_ids = tokenized.word_ids(batch_index=i)
        # 입력에 CLS/SEP 등의 None이 포함됨
        word_to_first_wp = {}
        seen = set()
        for idx, w in enumerate(word_ids):
            if w is None:
                continue
            if w not in seen:
                word_to_first_wp[w] = idx
                seen.add(w)

        # 단어 수
        num_words = max([-1] + [w for w in word_to_first_wp.keys()]) + 1 if word_to_first_wp else 0
        # 길이 정합성 검사 (토큰 수 == head/label 수)
        if num_words != len(heads) or num_words != len(deprels):
            # 길이가 어긋나면 가능한 범위까지 자릅니다 (데모 안전장치)
            cut = min(num_words, len(heads), len(deprels))
            heads = heads[:cut]
            deprels = deprels[:cut]

        # 첫 서브워드 표시
        word_starts = [0] * len(word_ids)
        for w, wp_idx in word_to_first_wp.items():
            word_starts[wp_idx] = 1

        # head 인덱스는 단어 단위(1-based). 0은 ROOT.
        # 학습 시에는 워드피스 인덱스 공간으로 매핑해야 하므로
        # ROOT는 -1로 보관, 이후 모델 내부에서 처리.
        mapped_heads = []
        mapped_deprels = []
        for dep_word_idx, (h, rel) in enumerate(zip(heads, deprels)):
            if dep_word_idx not in word_to_first_wp:
                continue
            dep_wp = word_to_first_wp[dep_word_idx]
            if h == 0:
                head_wp = -1  # ROOT
            else:
                hw = h - 1  # 1-based -> 0-based
                head_wp = word_to_first_wp.get(hw, -1)
            mapped_heads.append(head_wp)
            mapped_deprels.append(deprel_to_id.get(rel, deprel_to_id["root"]))

        batch_heads.append(mapped_heads)
        batch_deprels.append(mapped_deprels)
        batch_word_starts.append(word_starts)

    tokenized["heads"] = batch_heads
    tokenized["deprels"] = batch_deprels
    tokenized["word_starts"] = batch_word_starts
    return tokenized


class Biaffine(nn.Module):
    """biaffine = x^T U y + Wx + Vy + b (여기서는 텐서 연산 단순화 버전 사용)"""

    def __init__(self, in1: int, in2: int, out: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(out, in1, in2))
        self.W1 = nn.Linear(in1, out, bias=False)
        self.W2 = nn.Linear(in2, out, bias=True)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H1], y: [B, L, H2] -> [B, L, L, O]
        # 쌍별 bilinear 점수 + 선형항
        B, L, H1 = x.shape
        _, _, H2 = y.shape
        out = torch.einsum("blh,ohlm,bmH->blo", x, self.U, y.transpose(1, 2))  # [B, L, O]
        # 위 einsum은 단순화; 실제 [B,L,L,O] 생성을 위해 아래 구현 사용
        bilinear = torch.einsum("blo,ohlm->blm", torch.einsum("blh,ohl->blo", x, self.U.sum(dim=2)), y.transpose(1, 2))
        # 선형항
        w1 = self.W1(x)  # [B,L,O]
        w2 = self.W2(y)  # [B,L,O]
        # [B,L,L,O]
        scores = bilinear.unsqueeze(-1) + w1.unsqueeze(2) + w2.unsqueeze(1)
        return scores.squeeze(-1)


class RobertaBiaffineDependencyParser(nn.Module):
    """Transformer 인코더 위에 헤드/관계 예측을 위한 biaffine 모듈을 얹은 단순 파서."""

    def __init__(self, encoder_name: str, num_deprel: int, hidden_mlp: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size

        # 의존/지배 역할 분리용 MLP
        self.dep_mlp = nn.Sequential(nn.Linear(enc_dim, hidden_mlp), nn.ReLU())
        self.head_mlp = nn.Sequential(nn.Linear(enc_dim, hidden_mlp), nn.ReLU())

        # 헤드 점수 (각 토큰이 어떤 토큰을 head로 가질지)
        self.arc_biaffine = Biaffine(hidden_mlp, hidden_mlp, out=1)
        # 관계 라벨 점수 (의존, 지배 표현을 합쳐 라벨 분류)
        self.rel_biaffine = Biaffine(hidden_mlp, hidden_mlp, out=num_deprel)

    def forward(self, input_ids, attention_mask, heads=None, deprels=None, word_starts=None):
        # 인코더 출력
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,L,H]

        # 단어의 첫 서브워드 위치만 사용 (word_starts==1)
        if word_starts is not None:
            # mask: [B,L]
            mask = word_starts
        else:
            mask = attention_mask

        dep_h = self.dep_mlp(enc)
        head_h = self.head_mlp(enc)

        # arc scores: [B,L,L]
        arc_scores = self.arc_biaffine(dep_h, head_h)
        # 유효 위치 외 마스킹 (softmax에 큰 음수 가중치)
        minus_inf = -1e4
        # head 후보 마스크는 head_h가 유효한 위치
        head_mask = (mask > 0).unsqueeze(1).expand(-1, arc_scores.size(1), -1)
        arc_scores = arc_scores.masked_fill(~head_mask, minus_inf)

        outputs = {"arc_scores": arc_scores}

        loss = None
        if heads is not None and deprels is not None:
            # heads: List[List[int]] (워드피스 인덱스, -1은 ROOT)
            # deprels: List[List[int]]
            B, L = input_ids.size()
            device = input_ids.device

            # 각 문장마다 첫 서브워드 인덱스만 골라 유효 토큰 위치를 구합니다.
            valid_positions = (mask > 0)

            # 의존 토큰마다 소프트맥스 크로스엔트로피
            arc_logit = arc_scores  # [B,L,L]
            # gold head가 -1(=ROOT)인 경우, 가상의 ROOT 위치를 추가해 학습하는 것이 이상적이지만
            # 단순화를 위해 -1은 현재 문장 내 최대 길이의 자기 자신 위치로 대체하여 학습에서 제외
            target_heads = torch.full((B, L), fill_value=-100, dtype=torch.long, device=device)
            for b in range(B):
                # 유효 토큰 인덱스
                idxs = torch.nonzero(valid_positions[b], as_tuple=False).squeeze(-1).tolist()
                if not idxs:
                    continue
                gold_heads = heads[b]
                # 길이 안전장치
                gold_heads = gold_heads[: len(idxs)]
                for j, dep_wp in enumerate(idxs):
                    gh = gold_heads[j] if j < len(gold_heads) else -1
                    if gh >= 0 and gh < L:
                        target_heads[b, dep_wp] = gh
                    else:
                        target_heads[b, dep_wp] = -100  # 학습 제외

            arc_loss = F.cross_entropy(arc_logit.transpose(1, 2), target_heads, ignore_index=-100)

            # 라벨 예측: gold head 사용 (가능하면)
            # dep_h/ head_h에서 골드 head 위치를 인덱싱해 rel 점수 생성
            rel_logits_list = []
            rel_targets = []
            for b in range(B):
                idxs = torch.nonzero(valid_positions[b], as_tuple=False).squeeze(-1).tolist()
                if not idxs:
                    continue
                gold_heads = heads[b][: len(idxs)]
                gold_rels = deprels[b][: len(idxs)]
                for j, dep_wp in enumerate(idxs):
                    gh = gold_heads[j] if j < len(gold_heads) else -1
                    if gh < 0 or gh >= L:
                        continue
                    dep_vec = dep_h[b, dep_wp : dep_wp + 1]  # [1,H]
                    head_vec = head_h[b, gh : gh + 1]  # [1,H]
                    # [1,1,R]
                    rel_scores = self.rel_biaffine(dep_vec, head_vec).squeeze(0).squeeze(0)
                    rel_logits_list.append(rel_scores)
                    rel_targets.append(gold_rels[j])

            if rel_logits_list:
                rel_logits = torch.stack(rel_logits_list, dim=0)  # [N,R]
                rel_targets_t = torch.tensor(rel_targets, dtype=torch.long, device=device)
                rel_loss = F.cross_entropy(rel_logits, rel_targets_t)
            else:
                rel_loss = torch.tensor(0.0, device=device)

            loss = arc_loss + rel_loss
            outputs["loss"] = loss

        return outputs


def compute_metrics_builder():
    # 간단한 UAS/LAS(골드 라벨 기준) 근사 평가 (워드피스 첫 토큰 기준)
    def compute_metrics(p):
        # Trainer 기본 구조와 다르게 custom model의 출력 로짓을 바로 받기 어렵기 때문에
        # 여기서는 Trainer의 predictions 자리에 arc_scores를 전달하는 방식 대신
        # 평가 루프에서 evaluate가 제공하는 predictions를 head 인덱스로 변환했다고 가정합니다.
        # 간단화를 위해 accuracy만 보고, 자세한 평가는 별도 스크립트를 권장합니다.
        import numpy as np
        preds = p.predictions  # [B,L,L] 또는 변환된 [B,L]
        labels = p.label_ids  # [B,L]
        if preds.ndim == 3:
            preds = preds.argmax(-1)
        mask = labels != -100
        acc = (preds[mask] == labels[mask]).mean() if mask.any() else 0.0
        return {"head_accuracy": float(acc)}

    return compute_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune klue/roberta-large on KLUE-DP")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="예: '0' 또는 '0,1'")
    parser.add_argument("--device_index", type=int, default=0, help="VISIBLE 목록 내 사용 GPU 인덱스")
    parser.add_argument("--no_launch_blocking", action="store_true")
    parser.add_argument("--no_cuda_dsa", action="store_true")
    args = parser.parse_args()

    configure_logging()
    setup_cuda(
        cuda_visible_devices=args.cuda_visible_devices,
        device_index=args.device_index,
        enable_launch_blocking=not args.no_launch_blocking,
        enable_cuda_dsa=not args.no_cuda_dsa,
    )
    set_seed(args.seed)

    # 데이터 로드
    train, dev, test = load_klue_dp(args.data_dir)
    deprel_list = build_deprel_list((train, dev, test))
    deprel_to_id = {l: i for i, l in enumerate(deprel_list)}
    id_to_deprel = {i: l for l, i in deprel_to_id.items()}

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(dev),
            **({"test": Dataset.from_list(test)} if len(test) else {}),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(get_model_name())
    tokenized = ds.map(
        lambda ex: tokenize_and_align_for_dp(ex, tokenizer, deprel_to_id),
        batched=True,
        remove_columns=["tokens", "heads", "deprels"],
    )

    # 라벨 저장
    os.makedirs(args.output_dir, exist_ok=True)
    save_label_list(deprel_list, os.path.join(args.output_dir, "deprels.json"))

    model = RobertaBiaffineDependencyParser(get_model_name(), num_deprel=len(deprel_list))

    # 커스텀 데이터 콜레이터: 리스트 라벨을 텐서로 패딩
    def collate_fn(features: List[Dict]):
        batch = default_data_collator(
            [
                {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
                for f in features
            ]
        )
        # heads/deprels/word_starts 패딩
        def pad_list(list_of_lists, pad=-100):
            maxlen = max((len(x) for x in list_of_lists), default=0)
            return [x + [pad] * (maxlen - len(x)) for x in list_of_lists]

        batch["heads"] = torch.tensor(pad_list([f["heads"] for f in features]), dtype=torch.long)
        batch["deprels"] = torch.tensor(pad_list([f["deprels"] for f in features]), dtype=torch.long)
        batch["word_starts"] = torch.tensor(
            pad_list([f["word_starts"] for f in features], pad=0), dtype=torch.long
        )
        return batch

    # Trainer가 predictions/label_ids를 구성할 수 있도록 custom compute_metrics와 함께 사용
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="head_accuracy",
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        report_to=["none"],
    )

    # Trainer와의 호환을 위해 model의 forward를 감싸 predictions/label_ids를 구성
    class Wrapper(nn.Module):
        def __init__(self, base: RobertaBiaffineDependencyParser):
            super().__init__()
            self.base = base

        def forward(self, **batch):
            out = self.base(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                heads=batch.get("heads"),
                deprels=batch.get("deprels"),
                word_starts=batch.get("word_starts"),
            )
            # Trainer는 loss만 사용; predictions/label_ids는 compute_metrics 단계에서 사용됨
            return {"loss": out.get("loss", None), "arc_scores": out["arc_scores"]}

    wrapped = Wrapper(model)

    def compute_metrics_eval(eval_pred):
        # eval_pred.predictions 는 우리가 반환한 arc_scores 텐서
        arc_scores = eval_pred.predictions  # [B,L,L]
        # label_ids는 heads로 설정
        label_heads = eval_pred.label_ids  # [B,L]
        pred_heads = arc_scores.argmax(-1)
        mask = label_heads != -100
        acc = (pred_heads[mask] == label_heads[mask]).mean() if mask.any() else 0.0
        return {"head_accuracy": float(acc)}

    trainer = Trainer(
        model=wrapped,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_eval,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logging.getLogger(__name__).info("Eval metrics: %s", metrics)

    if "test" in tokenized:
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])  # type: ignore
        logging.getLogger(__name__).info("Test metrics: %s", test_metrics)

    # 최종 모델 저장
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    main()


