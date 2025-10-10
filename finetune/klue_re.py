"""
KLUE/roberta-large와 Hugging Face Transformers Trainer를 사용한 KLUE-RE 파인튜닝 스크립트.

이 스크립트는 datasets.load_dataset("klue", "re")로 KLUE 관계 추출(RE) 데이터셋을 로드하고,
주어/목적어 엔터티 스팬 주위에 마커를 삽입한 텍스트를 입력으로 사용하여
문장 분류(head)를 파인튜닝합니다. 평가에서는 정확도(accuracy)와 macro-F1을 보고합니다.

데이터셋 컬럼(KLUE/RE 하위 태스크):
- sentence: str – 두 개의 엔터티가 포함된 원문 문장
- subject_entity: dict – 주어 엔터티 정보 예시
  {"word": str, "start_idx": int, "end_idx": int, "type": str}
- object_entity: dict – 목적어 엔터티 정보(필드 동일)
- label: int – 관계 라벨 id. 라벨 이름은 features["label"].names에 존재
- id: Optional[str] – 예제 식별자

엔터티 타입을 강조하기 위해 특수 토큰을 추가합니다. 예:
  <SUBJ:PER> ... </SUBJ> 및 <OBJ:ORG> ... </OBJ>
이 토큰들은 tokenizer의 additional_special_tokens로 등록되어 하나의 토큰으로 처리됩니다.

참고 자료(References):
- KLUE benchmark: https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark
- KLUE dataset card: https://huggingface.co/datasets/klue/klue
- Model card (klue/roberta-large): https://huggingface.co/klue/roberta-large
- Transformers text classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
"""

from typing import Dict, Any, List, Tuple

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate
from utils.re_dataset_saver import save_tokenized_dataset_info

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 다른사람이 실수로 접속해서 메모리 초과 되서 끊기는 것 방지 가능
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = torch.device(f'cuda:0')  # VISIBLE DEVICES 중 0번째 사용할 시
torch.cuda.set_device(device)


SPECIAL_TOKENS = [
    "<SUBJ:", ">", "</SUBJ>", "<OBJ:", "</OBJ>",
]


def insert_entity_markers(
    sentence: str,
    subj: Dict[str, Any],
    obj_: Dict[str, Any],
) -> str:
    """
    - 문자 인덱스를 기준으로 엔터티 마커를 삽입합니다. 오프셋 변화를 고려해야 합니다.
    - 종료 태그는 엔터티 스팬의 끝 인덱스 뒤(포함)로, 시작 태그는 시작 인덱스 앞에 둡니다.
    - 예시 결과: "... <SUBJ:PER>홍길동</SUBJ> ... <OBJ:ORG>대한은행</OBJ> ..."
    """
    s_start, s_end = int(subj["start_idx"]), int(subj["end_idx"])  # KLUE에서 end는 포함(inclusive)
    o_start, o_end = int(obj_["start_idx"]), int(obj_["end_idx"])  # 포함(inclusive)

    # 단순 슬라이싱을 위해 주어 엔터티가 먼저 나오도록 보장합니다. 아니라면 순서를 교체합니다.
    first, second = ("subj", (s_start, s_end, subj)), ("obj", (o_start, o_end, obj_))
    if o_start < s_start:
        first, second = second, first

    def wrap(span_text: str, tag: str, ent_type: str) -> str:
        return f"<{tag}:{ent_type}>{span_text}</{tag}>"

    def apply(sentence_: str, start: int, end: int, tag: str, ent_type: str) -> Tuple[str, int]:
        # end는 포함(inclusive)이므로 파이썬 슬라이싱에선 end+1을 사용합니다.
        pre = sentence_[:start]
        mid = sentence_[start : end + 1]
        post = sentence_[end + 1 :]
        wrapped = wrap(mid, tag, ent_type)
        new_sentence = pre + wrapped + post
        delta = len(wrapped) - len(mid)
        return new_sentence, delta

    sent = sentence
    if first[0] == "subj":
        sent, delta = apply(sent, first[1][0], first[1][1], "SUBJ", str(first[1][2]["type"]))
        s_shift = delta
        # 첫 삽입으로 길이가 변했으므로 두 번째 엔터티 인덱스를 보정합니다.
        s2, e2 = second[1][0] + s_shift, second[1][1] + s_shift
        sent, _ = apply(sent, s2, e2, "OBJ", str(second[1][2]["type"]))
    else:
        sent, delta = apply(sent, first[1][0], first[1][1], "OBJ", str(first[1][2]["type"]))
        o_shift = delta
        s2, e2 = second[1][0] + o_shift, second[1][1] + o_shift
        sent, _ = apply(sent, s2, e2, "SUBJ", str(second[1][2]["type"]))

    return sent


def main() -> None:
    dataset = load_dataset("klue", "re")

    model_name = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 추가 마커 등록: 구분 토큰으로 취급되도록 구성합니다.
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    label_names: List[str] = dataset["train"].features["label"].names  # type: ignore[attr-defined]
    num_labels = len(label_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        attn_implementation="eager",  # 어텐션 맵 출력을 위해 eager 구현 사용
    )
    # 어텐션 맵 출력 활성화 (추론 시 output_attentions=True로도 제어 가능)
    model.config.output_attentions = True
    model.resize_token_embeddings(len(tokenizer))

    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [
            insert_entity_markers(s, se, oe)  # type: ignore[arg-type]
            for s, se, oe in zip(examples["sentence"], examples["subject_entity"], examples["object_entity"])
        ]
        enc = tokenizer(texts, truncation=True, return_attention_mask=True)
        enc["labels"] = examples["label"] # loss 계산, backprop 과정에서 사용될 라벨 지정
        return enc

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Insert entity markers and tokenize",
    )

    # 토크나이징된 데이터셋 내용을 파일로 저장
    save_tokenized_dataset_info(tokenized, tokenizer, label_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(p: Any) -> Dict[str, float]:
        preds = p.predictions
        if isinstance(preds, tuple): preds = preds[0]
        pred_labels = np.argmax(preds, axis=-1) # (batch_size, num_labels) -> (batch_size,)
        result = {
            "accuracy": float(acc.compute(predictions=pred_labels, references=p.label_ids)["accuracy"]),
            "f1_macro": float(f1.compute(predictions=pred_labels, references=p.label_ids, average="macro")["f1"]),
        }
        return result

    training_args = TrainingArguments(
        output_dir="./results/klue-re",
        logging_dir="./results/klue-re/logs",
        eval_strategy="epoch",  # evaluation_strategy → eval_strategy로 변경
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", tokenized.get("dev")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # trainer.train()
    # trainer.evaluate()


if __name__ == "__main__":
    main()


