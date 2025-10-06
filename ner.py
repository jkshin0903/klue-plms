"""
klue/roberta-large 기반 KLUE-NER 파인튜닝 스크립트.

입력 데이터 형식 (KLUE 공개 포맷과 호환):
- train.json, dev.json, test.json: 샘플 리스트
- 각 샘플: {"tokens": ["..."], "tags": ["B-PS", ...]} 형식이거나
  오리지널 KLUE 엔티티 포맷일 수 있으며, 내부적으로 tokens/tags로 정규화합니다.

실행 예시:
  python ner.py \
    --data_dir ./data/klue-ner \
    --output_dir ./outputs/ner \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    HAVE_SEQEVAL = True
except Exception:
    HAVE_SEQEVAL = False

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
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
)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_klue_ner(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """KLUE-NER 데이터셋을 로드하고 IOB2 토큰 라벨로 정규화합니다.

    파일이 이미 "tokens"/"tags"를 포함하면 그대로 사용합니다.
    오리지널 KLUE 포맷(문장+엔티티)인 경우, 공백 단위 토크나이즈 후 간단히 매핑합니다.
    (실서비스에선 문자 오프셋 기반 정밀 정렬을 권장합니다.)
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

    train = normalize(_read_json(os.path.join(data_dir, "train.json")))
    dev = normalize(_read_json(os.path.join(data_dir, "dev.json")))
    test_path = os.path.join(data_dir, "test.json")
    test = normalize(_read_json(test_path)) if os.path.exists(test_path) else []
    return train, dev, test


def build_label_list(datasets: Tuple[List[Dict], List[Dict], List[Dict]]) -> List[str]:
    labels = {"O"}
    for split in datasets:
        for s in split:
            labels.update(s["tags"])
    return sorted(labels)


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    # 단어 단위 토크나이즈 후 워드피스에 라벨을 정렬합니다.
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
    def compute_metrics(p):
        if not HAVE_SEQEVAL:
            # seqeval 미설치 시 토큰 단위 정확도로 대체
            preds = p.predictions.argmax(-1)
            labels = p.label_ids
            mask = labels != -100
            correct = (preds == labels) & mask
            acc = correct.sum() / mask.sum()
            return {"accuracy": float(acc)}

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

    return compute_metrics


def main():
    import argparse

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

    # Load and prepare data
    train, dev, test = load_klue_ner(args.data_dir)
    label_list = build_label_list((train, dev, test))
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(dev),
            **({"test": Dataset.from_list(test)} if len(test) else {}),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(get_model_name())
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
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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


