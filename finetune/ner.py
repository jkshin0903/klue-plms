"""
klue/roberta-large 기반 KLUE-NER 파인튜닝 스크립트.

데이터셋 구조(우선 포맷: TSV)
- 파일: train.tsv, dev.tsv, test.tsv (헤더 포함)
- 열 스키마:
  - tokens: 공백으로 구분된 토큰 리스트
    예) "이순신 은 위인 이다"
  - tags: 공백으로 구분된 IOB2 라벨 리스트(토큰 수와 동일 길이)
    예) "B-PER O O O O"
- 제약/불변식:
  - len(tokens) == len(tags)
  - 라벨은 IOB2 규칙을 따르며, "O" 라벨은 항상 포함됩니다.
- TSV 미존재 시 JSON(train.json, dev.json, test.json) / 원본(text, entities) fallback 지원

전처리 및 정렬(워드피스 기준)
- 토큰을 워드피스로 분할하고, 특수 토큰([CLS]/[SEP])은 -100으로 마스킹합니다.
- 서브워드는 이전 단어 라벨을 I-로 확장합니다(B-XXX → I-XXX 적용).
- 정렬 결과는 tokenizer 출력 + labels(-100 포함)로 반환합니다.

모델/학습 로직
- 모델: AutoModelForTokenClassification.from_pretrained
- 데이터 결합: DataCollatorForTokenClassification
- 지표: seqeval 설치 시 precision/recall/F1 및 classification_report 제공,
        미설치 시 토큰 정확도 대체
- EarlyStopping 콜백 사용 가능(예: patience=2)

훈련 플로우
1) 데이터 로드(TSV 우선) → 라벨 목록/맵 구성
2) DatasetDict로 변환 → map으로 토크나이즈/라벨 정렬 수행
3) 라벨 목록(labels.json) 저장, 모델/인자/콜레이터 생성
4) Trainer로 학습/평가(evaluation_strategy="epoch", metric_for_best_model="f1")
5) best model 저장 및(선택) test 평가
"""

from __future__ import annotations

import logging
import os
import argparse
from typing import Dict, List, Tuple

try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    HAVE_SEQEVAL = True
except Exception:
    HAVE_SEQEVAL = False

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from utils import (
    configure_logging,
    get_model_name,
    maybe_add_early_stopping,
    save_label_list,
    set_seed,
    setup_cuda,
    read_json,
    make_dataset_dict,
    build_id_maps,
    get_tokenizer,
    read_ner_tsv,
)


def load_klue_ner(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """KLUE-NER 데이터셋을 로드하고 IOB2 라벨로 정규화합니다.

    - 이미 tokens/tags가 있으면 그대로 사용합니다.
    - 원본 포맷(text/entities)은 공백 기준 토크나이즈 + 단순 문자열 매칭으로 IOB2를 구성합니다.
      정확한 라벨 정렬이 필요하면 문자 오프셋 기반 매핑을 사용해야 합니다.
    반환: (train, dev, test) 각 원소는 {"tokens", "tags"} 리스트
    """
    def normalize(samples: List[Dict]) -> List[Dict]:
        norm = []
        for s in samples:
            if "tokens" in s and "tags" in s:
                norm.append({"tokens": s["tokens"], "tags": s["tags"]})
                continue

            text = s.get("text", "")
            entities = s.get("entities", [])
            tokens = text.split()
            tags = ["O"] * len(tokens)
            # 문자열 매칭 기반의 단순 매핑 (데모 목적). 실제로는 문자 오프셋 사용 권장.
            for ent in entities:
                ent_text = ent.get("text")
                ent_type = ent.get("type", "ENT")
                for i, tok in enumerate(tokens):
                    if tok == ent_text:
                        tags[i] = f"B-{ent_type}"
            norm.append({"tokens": tokens, "tags": tags})
        return norm

    # TSV 우선, 없으면 JSON 사용
    train_tsv = read_ner_tsv(os.path.join(data_dir, "train.tsv"))
    dev_tsv = read_ner_tsv(os.path.join(data_dir, "dev.tsv"))
    test_tsv = read_ner_tsv(os.path.join(data_dir, "test.tsv"))

    if train_tsv and dev_tsv:
        train = train_tsv
        dev = dev_tsv
        test = test_tsv
    else:
        train = normalize(read_json(os.path.join(data_dir, "train.json")))
        dev = normalize(read_json(os.path.join(data_dir, "dev.json")))
        test_path = os.path.join(data_dir, "test.json")
        test = normalize(read_json(test_path)) if os.path.exists(test_path) else []
    return train, dev, test


def build_label_list(datasets: Tuple[List[Dict], List[Dict], List[Dict]]) -> List[str]:
    """세 분할(train/dev/test)의 라벨 합집합을 정렬해 반환합니다.

    - 테스트에만 등장하는 라벨 대응을 위해 전체 합집합을 사용합니다.
    - "O" 라벨은 항상 포함됩니다.
    """
    labels = {"O"}
    for split in datasets:
        for s in split:
            labels.update(s["tags"])
    return sorted(labels)


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    """단어 단위 라벨을 워드피스 토큰에 정렬합니다.

    규칙:
    - 특수 토큰은 -100으로 표시하여 손실에서 제외
    - 서브워드에는 이전 단어 라벨을 I- 접두로 확장
    - 라벨 ID 매핑은 `label_to_id`를 사용
    반환: tokenizer 출력 + "labels" 키가 포함된 dict
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        return_offsets_mapping=False,
    )
    aligned_labels = []
    for i, labels in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 특수 토큰은 손실 계산에서 무시
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[labels[word_idx]])
            else:
                # 서브워드의 경우, 기존 라벨이 B- 이면 I-로 바꿔서 이어붙입니다.
                label = labels[word_idx]
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                label_ids.append(label_to_id.get(label, label_to_id[labels[word_idx]]))
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics_builder(id_to_label):
    """Trainer의 compute_metrics 콜백을 생성합니다.

    - seqeval이 있으면 시퀀스 라벨링 지표를 반환
    - 없으면 토큰 정확도를 대체 지표로 사용
    """
    def _compute_metrics_seqeval(p, id_to_label):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        true_labels, true_preds = [], []
        for pred, lab in zip(preds, labels):
            curr_labs, curr_preds = [], []
            for p_id, l_id in zip(pred, lab):
                if l_id == -100:
                    continue
                curr_labs.append(id_to_label[int(l_id)])
                curr_preds.append(id_to_label[int(p_id)])
            true_labels.append(curr_labs)
            true_preds.append(curr_preds)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
            "report": classification_report(true_labels, true_preds),
        }

    def _compute_metrics_token_acc(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        mask = labels != -100
        correct = (preds == labels) & mask
        acc = correct.sum() / mask.sum()
        return {"accuracy": float(acc)}

    return (lambda p: _compute_metrics_seqeval(p, id_to_label)) if HAVE_SEQEVAL else _compute_metrics_token_acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune klue/roberta-large on KLUE-NER")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=5)
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

    # 데이터 로드 및 라벨 준비
    train, dev, test = load_klue_ner(args.data_dir)
    label_list = build_label_list((train, dev, test))
    label_to_id, id_to_label = build_id_maps(label_list)

    ds = make_dataset_dict(train, dev, test)

    tokenizer = get_tokenizer()
    tokenized = ds.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label_to_id),
        batched=True,
        remove_columns=["tokens", "tags"],
    )

    # Save labels for later use
    save_label_list(label_list, os.path.join(args.output_dir, "labels.json"))

    model = AutoModelForTokenClassification.from_pretrained(
        get_model_name(),
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",  # 학습 중 일정 step마다 로그 출력
        logging_steps=100,
        load_best_model_at_end=True,  # 가장 좋은 모델 자동 로드
        metric_for_best_model="f1",  # best model 기준 지표
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id_to_label),
        callbacks=maybe_add_early_stopping(patience=2),
    )

    trainer.train()
    metrics = trainer.evaluate()
    logging.getLogger(__name__).info("Eval metrics: %s", metrics)

    if "test" in tokenized:
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])  # type: ignore
        logging.getLogger(__name__).info("Test metrics: %s", test_metrics)

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()


