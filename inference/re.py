"""
KLUE-RE 추론 스크립트: 파인튜닝된 체크포인트를 로드하여 관계 라벨과 attention map을 반환합니다.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification

from utils import get_tokenizer, setup_cuda, insert_entity_markers


def predict(sentence: str, subject: Dict, obj: Dict, ckpt_dir: str):
    tokenizer = get_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir, output_attentions=True)
    model.eval()

    marked = insert_entity_markers(sentence, subject, obj)
    inputs = tokenizer(marked, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, C]
        attentions = outputs.attentions  # Tuple[Layer] of [1, heads, L, L]

    pred_id = logits.argmax(-1).item()
    # id2label은 체크포인트에 저장되어 있음
    id2label = model.config.id2label
    label = id2label.get(pred_id, str(pred_id))
    return {"label": label, "attentions": [a.squeeze(0).tolist() for a in attentions]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True, help='JSON: {"word":..., "type":...}')
    parser.add_argument("--object", type=str, required=True, help='JSON: {"word":..., "type":...}')
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()

    setup_cuda(args.cuda_visible_devices, args.device_index)

    subject = json.loads(args.subject)
    obj = json.loads(args.object)
    result = predict(args.sentence, subject, obj, args.ckpt_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


