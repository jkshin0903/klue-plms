"""
KLUE/roberta-large와 Hugging Face Transformers Trainer를 사용한 KLUE-NER 파인튜닝 스크립트.

이 스크립트는 datasets.load_dataset("klue", "ner")로 KLUE NER 데이터셋을 로드하고,
단어 단위 정렬을 유지한 채 토크나이즈하여 라벨을 서브워드에 올바르게 정렬한 뒤
토큰 분류(head)를 파인튜닝합니다. 평가 시 seqeval 지표를 보고합니다.

데이터셋 컬럼(KLUE/NER 하위 태스크):
- tokens: List[str] – 각 문장의 사전 토크나이즈된 토큰 리스트
- ner_tags: List[int] – IOB2 스타일의 NER 태그 id(토큰과 정렬). 라벨 이름은 features["ner_tags"].feature.names에 존재
- id: Optional[str] – 예제 식별자

중요 사항:
- is_split_into_words=True를 사용하여 토큰-단어 정렬을 유지하고, 단어 단위 라벨을 서브워드에 매핑합니다
  (첫 번째 서브워드에 라벨을 부여하고, 나머지는 -100으로 무시).
- 평가 지표는 evaluate.load("seqeval")을 통해 seqeval을 사용합니다.

참고 자료(References):
- KLUE benchmark: https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark
- KLUE dataset card: https://huggingface.co/datasets/klue/klue
- Model card (klue/roberta-large): https://huggingface.co/klue/roberta-large
- Transformers NER example: https://huggingface.co/docs/transformers/tasks/token_classification
- Seqeval metrics: https://github.com/chakki-works/seqeval
"""

# 라벨 마스킹 상수 (손실/평가에서 무시할 인덱스)
IGNORE_INDEX: int = -100

from typing import Dict, List, Any

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate


def main() -> None:
    # 데이터셋 로드
    dataset = load_dataset("klue", "ner")

    # 토크나이저 및 모델 준비
    model_name = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # NER 라벨 이름 목록
    label_names: List[str] = dataset["train"].features["ner_tags"].feature.names  # type: ignore[attr-defined]
    num_labels = len(label_names)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # 문자 단위 토큰에 맞춘 정렬/매핑: offset_mapping을 이용하여 서브워드 ↔ 원문 문자 위치를 정렬합니다.
    def tokenize_and_align_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
        # 문자 리스트를 그대로 이어붙여 원문 문장을 구성합니다(공백 포함)
        texts = ["".join(chars) for chars in examples["tokens"]]

        tokenized_inputs = tokenizer(
            texts,
            truncation=True,
            return_offsets_mapping=True,  # 각 서브워드가 덮는 원문 문자 구간(start, end)
            # 패딩은 data collator에서 처리하여 map 중 불필요한 패딩 계산을 피합니다
        )

        labels: List[List[int]] = []
        for sample_index, offsets in enumerate(tokenized_inputs["offset_mapping"]):
            # 문자 단위 라벨 시퀀스: 각 문자 인덱스에 대한 NER 태그 id
            char_labels: List[int] = examples["ner_tags"][sample_index]

            label_ids: List[int] = []
            last_token_start = None
            for (start, end) in offsets:
                # 특수 토큰([CLS]/[SEP] 등)은 (0,0) 또는 (0,0) 유사 패턴을 가짐 → 무시
                if start == end:
                    label_ids.append(IGNORE_INDEX)
                    continue

                # 이 서브워드가 커버하는 첫 문자 위치의 라벨을 사용
                # 동일 문자 구간에서 이어지는 서브워드(동일 start)는 -100 처리하여 중복 학습 방지
                if last_token_start is None or start != last_token_start:
                    # start가 문장 길이를 넘는 경우를 방지
                    label_ids.append(int(char_labels[start]) if 0 <= start < len(char_labels) else IGNORE_INDEX)
                    last_token_start = start
                else:
                    label_ids.append(IGNORE_INDEX)

            labels.append(label_ids)

        # 모델 입력에 labels 추가, offset_mapping은 더 이상 필요 없으므로 제거
        tokenized_inputs["labels"] = labels
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs

    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenize and align NER labels",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # seqeval 기반 평가 지표
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p: Any) -> Dict[str, float]:
        predictions = p.predictions
        if isinstance(predictions, tuple): predictions = predictions[0]
        preds = np.argmax(predictions, axis=2) # (batch_size, seq_len, num_labels) -> (batch_size, seq_len)

        # 예측/정답 id를 라벨 문자열로 변환
        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []
        for pred_row, label_row in zip(preds, p.label_ids):
            true_labels_row: List[str] = []
            true_preds_row: List[str] = []
            for p_id, l_id in zip(pred_row, label_row):
                if l_id == IGNORE_INDEX: continue
                true_labels_row.append(label_names[l_id])
                true_preds_row.append(label_names[p_id])
            true_labels.append(true_labels_row)
            true_predictions.append(true_preds_row)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        # 반환 키: precision, recall, f1, accuracy
        return {
            "precision": float(results.get("precision", 0.0)),
            "recall": float(results.get("recall", 0.0)),
            "f1": float(results.get("f1", 0.0)),
            "accuracy": float(results.get("accuracy", 0.0)),
        }

    training_args = TrainingArguments(
        output_dir="./results/klue-ner",
        logging_dir="./results/klue-ner/logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("dev")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()


