"""
KLUE-RE 추론 스크립트

개요:
- 파인튜닝된 문장 분류 모델을 로드하여 (문장, 주어, 목적어) 삼자 정보로 관계 라벨을 예측합니다.
- 레이어별 attention map을 함께 반환할 수 있습니다.

입력/출력:
- 입력: `--sentence` 문자열, `--subject`/`--object` JSON 문자열({"word", "type"}), 체크포인트 디렉터리.
- 출력 JSON:
  {
    "label": "...",
    "attentions": [[[...]]]  # 레이어 x 헤드 x L x L
  }
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification

from utils import get_tokenizer, setup_cuda, insert_entity_markers


def predict(sentence: str, subject: Dict, obj: Dict, ckpt_dir: str):
    """문장/엔티티 정보를 받아 관계 라벨을 예측합니다."""
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


