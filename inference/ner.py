"""
KLUE-NER 추론 스크립트: 파인튜닝된 체크포인트를 로드하여 문장 토큰에 대한 태깅과
모델 내부 attention map을 반환합니다.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from transformers import AutoModelForTokenClassification

from utils import get_tokenizer, setup_cuda, read_json


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def predict(tokens: List[str], ckpt_dir: str):
    tokenizer = get_tokenizer()
    labels = load_labels(f"{ckpt_dir}/labels.json")
    id_to_label = {i: l for i, l in enumerate(labels)}

    model = AutoModelForTokenClassification.from_pretrained(ckpt_dir, output_attentions=True)
    model.eval()

    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, L, C]
        attentions = outputs.attentions  # Tuple[Layer] of [1, heads, L, L]

    pred_ids = logits.argmax(-1).squeeze(0).tolist()
    pred_labels = [id_to_label[i] for i in pred_ids]
    return {"tokens": tokens, "labels": pred_labels, "attentions": [a.squeeze(0).tolist() for a in attentions]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokens", type=str, nargs="+", help="공백 분리 토큰 입력")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()

    setup_cuda(args.cuda_visible_devices, args.device_index)

    result = predict(args.tokens, args.ckpt_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


