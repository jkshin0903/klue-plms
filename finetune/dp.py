"""
klue/roberta-large 기반 KLUE-DP(의존 구문 분석) 파인튜닝 스크립트.

데이터셋 구조(간단 요약):
- train.json, dev.json, test.json 파일을 가정하며 각 파일은 샘플 리스트입니다.
- 각 샘플은 다음 키를 포함합니다.
  - "tokens": 공백 토큰 단위의 토큰 문자열 리스트
  - "heads": 각 단어의 head 단어 인덱스(1-based), 0은 ROOT를 의미
  - "deprels": 각 단어의 의존관계 라벨 문자열 리스트

전처리 개요:
- 토큰 리스트를 워드피스 단위로 토크나이즈합니다.
- 단어 첫 서브워드 위치에만 학습 신호를 주기 위해 `word_starts` 마스크를 생성합니다.
- 단어 단위 head/label을 첫 서브워드 인덱스로 매핑하여 워드피스 인덱스 공간으로 정렬합니다.

모델 개요:
- RoBERTa 인코더 위에 dep/head MLP 투영 후 biaffine 레이어로
  (1) arc(head 선택) 점수와 (2) 관계(deprel) 점수를 계산합니다.
- 학습 시 arc 손실과 관계 손실을 함께 최적화합니다.
"""

from __future__ import annotations
import logging
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from utils import (
    configure_logging,
    get_model_name,
    save_label_list,
    set_seed,
    setup_cuda,
    read_json,
    make_dataset_dict,
    get_tokenizer,
    read_dp_tsv,
)
from .models import Biaffine, RobertaBiaffineDependencyParser, Wrapper


def load_klue_dp(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """KLUE-DP 데이터셋을 로드합니다.

    입력 디렉터리에서 train/dev/test JSON을 읽고, 파서 학습에 필요한 키만 추립니다.
    존재하지 않을 수 있는 test.json은 없으면 빈 리스트를 반환합니다.
    반환되는 각 샘플은 {"tokens", "heads", "deprels"} 키만 포함합니다.
    """
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

    # TSV 우선, 없으면 JSON 사용
    train_tsv = read_dp_tsv(os.path.join(data_dir, "train.tsv"))
    dev_tsv = read_dp_tsv(os.path.join(data_dir, "dev.tsv"))
    test_tsv = read_dp_tsv(os.path.join(data_dir, "test.tsv"))

    if train_tsv and dev_tsv:
        train = train_tsv
        dev = dev_tsv
        test = test_tsv
    else:
        train = pick(read_json(os.path.join(data_dir, "train.json")))
        dev = pick(read_json(os.path.join(data_dir, "dev.json")))
        test_path = os.path.join(data_dir, "test.json")
        test = pick(read_json(test_path)) if os.path.exists(test_path) else []
    return train, dev, test


def build_deprel_list(datasets: Tuple[List[Dict], List[Dict], List[Dict]]) -> List[str]:
    """훈련/검증/테스트 전체에서 등장하는 의존관계 라벨 목록을 정렬해 반환합니다.

    - "root" 라벨은 최소 포함되어야 하므로 기본으로 추가합니다.
    - 테스트에만 등장하는 라벨 대비를 위해 전체 합집합을 사용합니다.
    """
    labels = set(["root"])  # 기본값 포함
    for split in datasets:
        for s in split:
            labels.update(s.get("deprels", []))
    return sorted(labels)


def tokenize_and_align_for_dp(examples, tokenizer, deprel_to_id):
    """단어 단위 어노테이션을 워드피스 토큰 인덱스에 정렬합니다.

    정렬 규칙:
    - 각 단어의 첫 서브워드 위치에만 학습 신호를 부여(`word_starts`=1)
    - head 인덱스(1-based)는 해당 head 단어의 첫 서브워드 인덱스로 매핑
    - ROOT(0)는 -1로 보관하여 유효 head가 아님을 표시하며 손실 계산에서 무시됩니다.
    - 문장 또는 라벨 길이 불일치 시(잘린 경우) 공통 길이에 맞춰 자릅니다.
    반환 항목:
    - input_ids/attention_mask(Transformers 표준)
    - heads: 워드피스 기준 head 인덱스 리스트들의 리스트, 패딩은 collate에서 수행
    - deprels: 의존관계 라벨 ID 리스트들의 리스트
    - word_starts: 각 시퀀스 길이와 동일한 0/1 마스크(첫 서브워드=1)
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        return_offsets_mapping=False,
    )

    batch_heads = []
    batch_deprels = []
    batch_word_starts = []

    for i, (heads, deprels) in enumerate(zip(examples["heads"], examples["deprels"])):
        word_ids = tokenized.word_ids(batch_index=i)
        word_to_first_wp = {}
        seen = set()
        for idx, w in enumerate(word_ids):
            if w is None:
                continue
            if w not in seen:
                word_to_first_wp[w] = idx
                seen.add(w)

        # 워드 수가 라벨 길이와 다를 수 있으므로 안전하게 공통 부분으로 자릅니다.
        num_words = max([-1] + [w for w in word_to_first_wp.keys()]) + 1 if word_to_first_wp else 0
        if num_words != len(heads) or num_words != len(deprels):
            cut = min(num_words, len(heads), len(deprels))
            heads = heads[:cut]
            deprels = deprels[:cut]

        word_starts = [0] * len(word_ids)
        for w, wp_idx in word_to_first_wp.items():
            word_starts[wp_idx] = 1

        mapped_heads = []
        mapped_deprels = []
        for dep_word_idx, (h, rel) in enumerate(zip(heads, deprels)):
            if dep_word_idx not in word_to_first_wp:
                continue
            dep_wp = word_to_first_wp[dep_word_idx]
            if h == 0:
                head_wp = -1  # ROOT는 -1로 마킹하여 유효 head 아님을 표시
            else:
                hw = h - 1
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


from .models import Biaffine, RobertaBiaffineDependencyParser, Wrapper


def collate_fn(features: List[Dict]):
    """리스트 기반 정렬 결과를 패딩하여 텐서 배치로 변환합니다.

    - heads/deprels: 길이 불일치가 있으므로 -100을 패딩 값으로 사용
    - word_starts: 마스크이므로 0으로 패딩
    """
    batch = default_data_collator(
        [
            {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
            for f in features
        ]
    )
    def pad_list(list_of_lists, pad=-100):
        maxlen = max((len(x) for x in list_of_lists), default=0)
        return [x + [pad] * (maxlen - len(x)) for x in list_of_lists]

    batch["heads"] = torch.tensor(pad_list([f["heads"] for f in features]), dtype=torch.long)
    batch["deprels"] = torch.tensor(pad_list([f["deprels"] for f in features]), dtype=torch.long)
    batch["word_starts"] = torch.tensor(
        pad_list([f["word_starts"] for f in features], pad=0), dtype=torch.long
    )
    return batch


def compute_metrics_eval(eval_pred):
    """arc(head) 정확도 계산.

    - 레이블의 -100은 무시
    - arc_scores argmax로 예측 head를 얻어 비교
    """
    arc_scores = eval_pred.predictions
    label_heads = eval_pred.label_ids
    pred_heads = arc_scores.argmax(-1)
    mask = label_heads != -100
    acc = (pred_heads[mask] == label_heads[mask]).mean() if mask.any() else 0.0
    return {"head_accuracy": float(acc)}


def main():
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

    train, dev, test = load_klue_dp(args.data_dir)
    deprel_list = build_deprel_list((train, dev, test))
    deprel_to_id = {l: i for i, l in enumerate(deprel_list)}

    ds = make_dataset_dict(train, dev, test)

    tokenizer = get_tokenizer()
    tokenized = ds.map(
        lambda ex: tokenize_and_align_for_dp(ex, tokenizer, deprel_to_id),
        batched=True,
        remove_columns=["tokens", "heads", "deprels"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_label_list(deprel_list, os.path.join(args.output_dir, "deprels.json"))

    model = RobertaBiaffineDependencyParser(get_model_name(), num_deprel=len(deprel_list))

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

    wrapped = Wrapper(model)

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

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    main()


