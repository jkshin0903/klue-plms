"""
klue/roberta-large 기반 KLUE-RE(문장 관계 추출) 파인튜닝 스크립트.

입력 데이터 형식 (KLUE 공개 포맷과 호환):
- train.json, dev.json, test.json: 샘플 리스트
- 각 샘플 예시:
  {
    "sentence": "문장 텍스트",
    "subject_entity": {"word": "김철수", "type": "PER"},
    "object_entity": {"word": "삼성", "type": "ORG"},
    "label": "org:founded_by"
  }

간단한 전처리 방법:
- 문장 내의 주어/목적어 엔티티를 특수 마커로 감싸 모델이 관계 단서를 잘 보도록 합니다.
  예: [SUBJ-PER] 김철수 [/SUBJ-PER] 와 [OBJ-ORG] 삼성 [/OBJ-ORG]

실행 예시:
  python re.py \
    --data_dir ./data/klue-re \
    --output_dir ./outputs/re \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils import configure_logging, get_model_name, save_label_list, set_seed, setup_cuda


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_klue_re(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """KLUE-RE 데이터셋을 로드하고 필요한 키만 남겨 반환합니다."""
    def pick(samples: List[Dict]) -> List[Dict]:
        out = []
        for s in samples:
            out.append(
                {
                    "sentence": s.get("sentence", ""),
                    "subject": s.get("subject_entity", {}),
                    "object": s.get("object_entity", {}),
                    "label": s.get("label", "no_relation"),
                }
            )
        return out

    train = pick(_read_json(os.path.join(data_dir, "train.json")))
    dev = pick(_read_json(os.path.join(data_dir, "dev.json")))
    test_path = os.path.join(data_dir, "test.json")
    test = pick(_read_json(test_path)) if os.path.exists(test_path) else []
    return train, dev, test


def build_label_list(datasets: Tuple[List[Dict], List[Dict], List[Dict]]) -> List[str]:
    labels = set()
    for split in datasets:
        for s in split:
            labels.add(s["label"])
    # 테스트에만 등장하는 라벨 대비를 위해 no_relation 기본값 포함
    labels.add("no_relation")
    return sorted(labels)


def insert_entity_markers(sentence: str, subject: Dict, obj: Dict) -> str:
    """문장에 엔티티 마커를 삽입합니다. (단순 문자열 치환 기반)

    - 실제 KLUE 원본은 문자 오프셋이 있으므로 오프셋 기반 삽입이 바람직합니다.
    - 여기서는 word 필드가 문장 내에 그대로 등장한다고 가정합니다.
    """
    s_word = subject.get("word", "")
    s_type = subject.get("type", "ENT")
    o_word = obj.get("word", "")
    o_type = obj.get("type", "ENT")

    # 충돌을 줄이기 위해 길이가 긴 엔티티부터 치환
    pairs = sorted(
        [(s_word, f"[SUBJ-{s_type}] {s_word} [/SUBJ-{s_type}]"), (o_word, f"[OBJ-{o_type}] {o_word} [/OBJ-{o_type}]")],
        key=lambda x: len(x[0]),
        reverse=True,
    )
    out = sentence
    for src, rep in pairs:
        if src:
            out = out.replace(src, rep, 1)
    return out


def preprocess_examples(examples, tokenizer, label_to_id):
    # 문장에 엔티티 마커를 삽입한 후 일반 텍스트 분류 입력으로 토크나이즈
    marked_texts = [
        insert_entity_markers(sen, sub, obj)
        for sen, sub, obj in zip(examples["sentence"], examples["subject"], examples["object"])
    ]
    model_inputs = tokenizer(marked_texts, truncation=True, padding=False)
    model_inputs["labels"] = [label_to_id[l] for l in examples["label"]]
    return model_inputs


def compute_metrics_builder(id_to_label):
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        accuracy = (preds == labels).mean()
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}

    return compute_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune klue/roberta-large on KLUE-RE")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
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
    train, dev, test = load_klue_re(args.data_dir)
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
        lambda ex: preprocess_examples(ex, tokenizer, label_to_id),
        batched=True,
        remove_columns=["sentence", "subject", "object", "label"],
    )

    # 라벨 저장
    os.makedirs(args.output_dir, exist_ok=True)
    save_label_list(label_list, os.path.join(args.output_dir, "labels.json"))

    model = AutoModelForSequenceClassification.from_pretrained(
        get_model_name(),
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

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
        metric_for_best_model="macro_f1",
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
        compute_metrics=compute_metrics_builder(id_to_label),
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


