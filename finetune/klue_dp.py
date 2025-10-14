"""
klue/roberta-base 위에 바이어파인(biaffine) 의존 구문 분석기 헤드를 얹은 KLUE-DP 파인튜닝 스크립트.

이 스크립트는 datasets.load_dataset("klue", "dp")로 KLUE 의존 구문 분석(DP) 데이터셋을 로드하고,
단어 정렬을 유지한 토크나이징을 수행한 뒤 각 토큰의 head와 의존관계를 예측하는
바이어파인 파서를 학습합니다.

데이터셋 컬럼(KLUE/DP 하위 태스크):
- tokens: List[str] – 문장 단위의 사전 토크나이즈된 토큰(공백/형태소 기준)
- pos: List[str] – 각 토큰의 품사(POS) 태그(모델에서 반드시 사용하지는 않음)
- head: List[int] – 단어 단위 head 인덱스(데이터셋에 따라 0/1 기반; KLUE는 ROOT head로 0 사용)
- deprel: List[str] 또는 ClassLabel – 각 토큰의 의존관계 라벨; features["deprel"].feature.names로 접근 가능
- id: Optional[str] – 예제 식별자

구현 세부사항:
- 인코더 출력 위에 표준 바이어파인 파서(Dozat and Manning, 2017)를 적용합니다.
- 워드피스 정렬: 각 단어의 첫 번째 서브토큰에만 라벨을 할당하고 나머지는 -100으로 마스킹합니다.
- Arc 예측: 토큰 간 쌍별 점수 s_ij를 계산하고, 토큰 i의 정답 head 인덱스 j에 대해 교차 엔트로피를 최적화합니다.
- Relation 예측: 정답 head-의존어 쌍에 대한 라벨 점수를 바이어파인 분류기로 계산합니다.

참고 자료(References):
- KLUE benchmark: https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark
- KLUE dataset card: https://huggingface.co/datasets/klue/klue
- Model card (klue/roberta-base): https://huggingface.co/klue/roberta-base
- Biaffine parser: https://arxiv.org/abs/1611.01734
- Transformers custom heads: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel
"""

# 라벨 마스킹 상수 (손실/평가에서 무시할 인덱스)
IGNORE_INDEX: int = -100

from typing import Dict, Any, List, Tuple

import os
# import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate

# 프로젝트 루트 디렉토리를 Python 경로에 추가
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.dp_dataset_saver import save_tokenized_dataset_info

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 다른사람이 실수로 접속해서 메모리 초과 되서 끊기는 것 방지 가능
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = torch.device(f'cuda:0')  # VISIBLE DEVICES 중 0번째 사용할 시
torch.cuda.set_device(device)


def pick_rel_logits_for_heads(rel_scores: torch.Tensor, head_idx: torch.Tensor) -> torch.Tensor:
    """고급 인덱싱으로 (B,L,L,R)에서 각 토큰의 head 위치(j)에 해당하는 (B,L,R) 관계 로짓을 선택합니다.

    - rel_scores: (batch_size, seq_len, seq_len, num_relations)
    - head_idx: (batch_size, seq_len), 토큰별 head 인덱스. 음수는 0으로 클램프(CLS/ROOT) 처리됩니다.
    반환: (batch_size, seq_len, num_relations)
    """
    batch_size, seq_len, _, num_relations = rel_scores.shape
    dev = rel_scores.device
    batch_idx = torch.arange(batch_size, device=dev).unsqueeze(1).expand(batch_size, seq_len) # (batch_size,) -> (batch_size, 1) -> (batch_size, seq_len)
    dep_idx = torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, seq_len) # (seq_len,) -> (1, seq_len) -> (batch_size, seq_len)
    head_idx_clamped = head_idx.clamp(min=0)
    return rel_scores[batch_idx, dep_idx, head_idx_clamped, :]

class Biaffine(nn.Module):
    """쌍별 상호작용을 스코어링하기 위한 바이어파인 변환 레이어.

    입력 텐서 형태:
    - dep: (batch_size, seq_len, dep_hidden_dim)
    - head: (batch_size, seq_len, head_hidden_dim)

    출력 텐서 형태:
    - out_features > 1 (관계 라벨 점수 등): (batch_size, seq_len, seq_len, out_features)
    - out_features == 1 (아크 점수 등): (batch_size, seq_len, seq_len)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        bias_dep: bool = True,
        bias_head: bool = True,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.bias_dep = bias_dep
        self.bias_head = bias_head
        self.weight = nn.Parameter(torch.zeros(out_features, in_features + int(bias_dep), in_features + int(bias_head)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, dep: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        """
        - dep: (batch_size, seq_len, dep_hidden_dim)
        - head: (batch_size, seq_len, head_hidden_dim)
        - W: (out_features, dep_hidden_dim_with_bias, head_hidden_dim_with_bias)
        - out: (batch_size, out_features, seq_len, seq_len)
            - out_features > 1이면 (batch_size, seq_len, seq_len, out_features)로 전치, 아니면 (batch_size, seq_len, seq_len)
        """
        if self.bias_dep:
            ones = torch.ones_like(dep[..., :1])
            dep = torch.cat([dep, ones], dim=-1)
        if self.bias_head:
            ones = torch.ones_like(head[..., :1])
            head = torch.cat([head, ones], dim=-1)
        # einsum으로 dep @ W @ head^T 계산
        s = torch.einsum("bxi, oij, byj -> boxy", dep, self.weight, head)
        if self.out_features == 1:
            return s.squeeze(1)  # (batch_size, seq_len, seq_len)
        return s.permute(0, 2, 3, 1)  # (batch_size, seq_len, seq_len, out_features)


class BiaffineParserConfig(PretrainedConfig):
    """바이어파인 의존 구문 분석(Biaffine Dependency Parsing)을 위한 설정 클래스.

    기능:
    - Hugging Face `PretrainedConfig`를 상속하여 백본 인코더 이름과 바이어파인 헤드의 하이퍼파라미터를 관리합니다.
    - `RobertaBiaffineDependencyParser`가 MLP 차원과 관계 라벨 수 등을 초기화하는 데 사용됩니다.

    생성자 인자:
    - encoder_name (str): 사용할 사전학습 인코더 모델 이름/경로. 기본값 "klue/roberta-base".
    - mlp_arc (int): 아크(헤드-의존어) 점수 산출을 위한 MLP 은닉 차원. 기본값 512.
    - mlp_rel (int): 관계 라벨 점수 산출을 위한 MLP 은닉 차원. 기본값 256.
    - num_relations (int): 의존관계 라벨의 개수(데이터셋의 deprel 클래스 수). 기본값 40.
    - **kwargs: `PretrainedConfig` 공통 옵션 전달용(예: id2label, label2id 등).
    """
    model_type = "biaffine-parser"

    def __init__(
        self,
        encoder_name: str = "klue/roberta-base",
        mlp_arc: int = 512,
        mlp_rel: int = 256,
        num_relations: int = 40,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.mlp_arc = mlp_arc
        self.mlp_rel = mlp_rel
        self.num_relations = num_relations
        # 기본적으로 어텐션 맵을 출력하도록 설정
        # self.output_attentions = True


class RobertaBiaffineDependencyParser(PreTrainedModel):
    """로버타 인코더 위에 바이어파인 아크/관계 헤드를 얹은 의존 구문 분석 모델.

    기능 개요:
    - 인코더 은닉표현으로부터 모든 토큰 쌍의 아크 점수(헤드-의존어)와 관계 라벨별 점수를 계산합니다.

    주요 입출력 텐서 형태:
    - 인코더 출력(enc_out): (batch_size, seq_len, hidden_size)
    - 아크 투영(dep_arc/head_arc): (batch_size, seq_len, arc_hidden_dim)
    - 관계 투영(dep_rel/head_rel): (batch_size, seq_len, rel_hidden_dim)
    - 아크 로짓(arc_scores): (batch_size, seq_len, seq_len)
    - 관계 로짓(rel_scores): (batch_size, seq_len, seq_len, num_relations)
    - 학습 시 라벨(labels_head, labels_deprel): (batch_size, seq_len)
    """
    config_class = BiaffineParserConfig

    def __init__(self, config: BiaffineParserConfig) -> None:
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden = self.encoder.config.hidden_size

        self.mlp_dep_arc = nn.Sequential(nn.Linear(hidden, config.mlp_arc), nn.ReLU(), nn.Dropout(0.33))
        self.mlp_head_arc = nn.Sequential(nn.Linear(hidden, config.mlp_arc), nn.ReLU(), nn.Dropout(0.33))
        self.mlp_dep_rel = nn.Sequential(nn.Linear(hidden, config.mlp_rel), nn.ReLU(), nn.Dropout(0.33))
        self.mlp_head_rel = nn.Sequential(nn.Linear(hidden, config.mlp_rel), nn.ReLU(), nn.Dropout(0.33))

        self.arc_biaffine = Biaffine(config.mlp_arc, out_features=1)
        self.rel_biaffine = Biaffine(config.mlp_rel, out_features=config.num_relations)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_head: torch.Tensor = None,
        labels_deprel: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        입력:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - labels_head: (batch_size, seq_len) | 선택, 각 토큰의 정답 head 인덱스(없으면 추론)
        - labels_deprel: (batch_size, seq_len) | 선택, 각 토큰의 정답 의존관계 라벨 id

        출력:
        - "arc_logits": (batch_size, seq_len, seq_len),
        - "rel_logits": (batch_size, seq_len, seq_len, num_relations),
        - "loss": 스칼라 | 학습 시 제공
        """
        # 하위 인코더에 어텐션 출력 플래그를 전달
        return_attn = self.config.output_attentions
        enc_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attn,
        )
        enc_out = enc_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        dep_arc = self.mlp_dep_arc(enc_out)  # (batch_size, seq_len, arc_hidden_dim)
        head_arc = self.mlp_head_arc(enc_out)  # (batch_size, seq_len, arc_hidden_dim)
        arc_scores = self.arc_biaffine(dep_arc, head_arc)  # (batch_size, seq_len, seq_len)

        dep_rel = self.mlp_dep_rel(enc_out)  # (batch_size, seq_len, rel_hidden_dim)
        head_rel = self.mlp_head_rel(enc_out)  # (batch_size, seq_len, rel_hidden_dim)
        rel_scores = self.rel_biaffine(dep_rel, head_rel)  # (batch_size, seq_len, seq_len, num_relations)

        output: Dict[str, torch.Tensor] = {"arc_logits": arc_scores, "rel_logits": rel_scores}
        if return_attn and hasattr(enc_outputs, "attentions") and enc_outputs.attentions is not None:
            # Tuple[batch, num_heads, seq_len, seq_len] per layer
            output["attentions"] = enc_outputs.attentions

        if labels_head is not None and labels_deprel is not None:
            # Arc 손실: 각 토큰 i에 대해 head 인덱스 j를 예측하는 교차 엔트로피
            batch_size, seq_len, _ = arc_scores.size()
            arc_logits = arc_scores  # (batch_size, seq_len, seq_len)
            arc_loss = F.cross_entropy(
                arc_logits.reshape(batch_size * seq_len, seq_len),
                labels_head.reshape(batch_size * seq_len),
                ignore_index=IGNORE_INDEX,
            )

            # Relation 손실: 정답 head 위치의 스코어를 선택해 라벨 분류 수행(고급 인덱싱 사용)
            rel_logits_gold = pick_rel_logits_for_heads(rel_scores, labels_head)
            rel_loss = F.cross_entropy(
                rel_logits_gold.reshape(batch_size * seq_len, -1),
                labels_deprel.reshape(batch_size * seq_len),
                ignore_index=IGNORE_INDEX,
            )

            loss = arc_loss + rel_loss
            output["loss"] = loss

        return output


def build_labels_and_align(
    tokenizer: Any,
    words: List[str],
    heads: List[int],
    deprels: List[str],
    deprel_to_id: Dict[str, int],
) -> Dict[str, Any]:
    enc = tokenizer(
        words,
        truncation=True,
        is_split_into_words=True,
        return_attention_mask=True,
    )
    word_ids = enc.word_ids() # 개별 토큰이 속한 단어의 인덱스(0-based)

    # 단어 인덱스 -> 첫 서브토큰 인덱스 매핑
    word_to_first_token: Dict[int, int] = {}
    for i, w in enumerate(word_ids):
        if w is None: continue # 특수 토큰(CLS/SEP), 패딩 토큰 무시
        if w not in word_to_first_token:
            word_to_first_token[w] = i

    seq_len = len(enc["input_ids"])
    labels_head = [IGNORE_INDEX] * seq_len
    labels_deprel = [IGNORE_INDEX] * seq_len

    # 각 단어의 첫 서브토큰에만 라벨을 부여하고, 단어 단위 head를 서브토큰 인덱스로 변환
    for w_idx, t_idx in word_to_first_token.items():
        gold_head_word = int(heads[w_idx])
        # KLUE-DP: ROOT는 0, 나머지는 1부터 시작
        if gold_head_word == 0:  # ROOT
            gold_head_tok = 0  # CLS 토큰으로 매핑
        else:
            # head 인덱스를 0-based로 변환하여 매핑 (1-based → 0-based)
            head_word_idx = gold_head_word - 1
            gold_head_tok = word_to_first_token.get(head_word_idx, 0)
        labels_head[t_idx] = gold_head_tok
        labels_deprel[t_idx] = deprel_to_id[deprels[w_idx]]

    enc["labels_head"] = labels_head
    enc["labels_deprel"] = labels_deprel
    return enc


def main() -> None:
    dataset = load_dataset("klue", "dp")

    # 의존관계 라벨 이름/개수 추출
    rel_feature = dataset["train"].features["deprel"]
    print(f"Deprel feature type: {type(rel_feature)}")
    print(f"Deprel feature: {rel_feature}")
    
    # deprel이 Value 타입인 경우 고유값들을 추출
    if hasattr(rel_feature, 'names'):
        num_relations = len(rel_feature.names)
        rel_names = rel_feature.names
    else:
        # Value 타입인 경우 모든 고유값을 수집 (train + validation)
        all_deprels = []
        for split in ["train", "validation"]:
            if split in dataset:
                for sample in dataset[split]:
                    all_deprels.extend(sample["deprel"])
        unique_deprels = sorted(list(set(all_deprels)))
        num_relations = len(unique_deprels)
        rel_names = unique_deprels
        print(f"Unique deprel values: {unique_deprels}")
    
    # 문자열 라벨을 정수 ID로 매핑하는 딕셔너리 생성
    deprel_to_id = {label: idx for idx, label in enumerate(rel_names)}
    print(f"Deprel to ID mapping: {deprel_to_id}")

    model_name = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    config = BiaffineParserConfig(encoder_name=model_name, num_relations=num_relations)
    model = RobertaBiaffineDependencyParser(config)

    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        batch_enc: Dict[str, List[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels_head": [],
            "labels_deprel": []
        }
        for words, heads, deprels in zip(examples["word_form"], examples["head"], examples["deprel"]):
            enc = build_labels_and_align(tokenizer, words, heads, deprels, deprel_to_id)
            batch_enc["input_ids"].append(enc["input_ids"])
            batch_enc["attention_mask"].append(enc["attention_mask"])
            batch_enc["labels_head"].append(enc["labels_head"])
            batch_enc["labels_deprel"].append(enc["labels_deprel"])
        return batch_enc

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenize tokens and align DP labels",
    )

    # 토크나이징된 데이터셋 내용을 파일로 저장
    # save_tokenized_dataset_info(tokenized, tokenizer, rel_names)

    class CustomDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, features):
            # input_ids와 attention_mask를 함께 패딩 처리
            batch = {}
            padded = self.tokenizer.pad(
                [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
                padding=True,
                return_tensors="pt"
            )
            batch["input_ids"] = padded["input_ids"]
            batch["attention_mask"] = padded["attention_mask"]
            
            # labels_head와 labels_deprel은 수동으로 패딩 처리
            max_len = max(len(f["labels_head"]) for f in features)
            batch["labels_head"] = torch.tensor([
                f["labels_head"] + [IGNORE_INDEX] * (max_len - len(f["labels_head"]))
                for f in features
            ])
            batch["labels_deprel"] = torch.tensor([
                f["labels_deprel"] + [IGNORE_INDEX] * (max_len - len(f["labels_deprel"]))
                for f in features
            ])
            
            return batch

    data_collator = CustomDataCollator(tokenizer)

    uas = evaluate.load("accuracy")  # proxy for UAS using head index equality
    las = evaluate.load("accuracy")  # proxy for LAS using both head and label equality

    def compute_metrics(p: Any) -> Dict[str, float]:
        arc_logits, rel_logits = p.predictions
        arc_pred = arc_logits.argmax(-1)
        # 관계 라벨 예측: 예측된 head 위치에서 라벨 로짓 선택(고급 인덱싱 사용)
        rel_logits_at_head = pick_rel_logits_for_heads(rel_logits, arc_pred)
        rel_pred = rel_logits_at_head.argmax(-1)

        mask = (p.label_ids["labels_head"] != IGNORE_INDEX)
        gold_head, gold_rel = p.label_ids["labels_head"], p.label_ids["labels_deprel"]

        # 원소 수준 마스크를 적용하여 1차원으로 평탄화하고 지표를 계산합니다.
        pred_head_f, head_gold_f = pred_head[mask], gold_head[mask]
        uas_val = float(uas.compute(predictions=pred_head_f, references=head_gold_f)["accuracy"])

        # 정식 LAS: head와 관계 라벨이 모두 일치해야 정답으로 간주
        both_correct = (pred_head == gold_head) & (rel_pred == gold_rel)
        # 원소 수준 마스크를 적용하여 1차원으로 평탄화하고 지표를 계산합니다.
        both_correct_f = both_correct[mask]
        las_val = float(las.compute(predictions=both_correct_f, references=[True] * both_correct_f.shape[0])["accuracy"])
        return {"uas": uas_val, "las": las_val}

    training_args = TrainingArguments(
        output_dir="./results/klue-dp",
        logging_dir="./results/klue-dp/logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        load_best_model_at_end=True,
        report_to=["none"],
    )

    # 특수 키를 가진 라벨 텐서를 전달하기 위한 커스텀 Trainer
    class DPTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
            labels_head = inputs.pop("labels_head")
            labels_deprel = inputs.pop("labels_deprel")
            outputs = model(**inputs, labels_head=labels_head, labels_deprel=labels_deprel)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):  # type: ignore[override]
            labels_head = inputs.get("labels_head")
            labels_deprel = inputs.get("labels_deprel")
            has_labels = labels_head is not None and labels_deprel is not None

            # 분리된 라벨은 별도로 보관하고, 나머지 입력으로 모델 호출
            inputs_no_labels = {k: v for k, v in inputs.items() if k not in ("labels_head", "labels_deprel")}

            with torch.no_grad():
                outputs = model(
                    **inputs_no_labels,
                    labels_head=labels_head if has_labels else None,
                    labels_deprel=labels_deprel if has_labels else None,
                )

            loss = outputs.get("loss") if has_labels else None
            logits = (outputs["arc_logits"], outputs["rel_logits"])  # compute_metrics에서 사용

            if prediction_loss_only: return (loss, None, None)

            label_pack = {"labels_head": labels_head, "labels_deprel": labels_deprel} 
            return (loss, logits, label_pack if has_labels else None)

    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", tokenized.get("dev")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()


