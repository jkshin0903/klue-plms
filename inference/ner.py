"""
KLUE-NER 추론 스크립트

개요:
- 파인튜닝된 토큰 분류 모델을 로드하여 입력 토큰 시퀀스의 IOB 태그를 예측합니다.
- 모델이 출력하는 레이어별 attention map을 함께 반환할 수 있습니다.

입력(우선 포맷: TSV):
- `--input_tsv path.tsv`로 헤더가 있는 TSV를 입력하면 행별로 tokens 열을 읽어 일괄 추론합니다.
- 또는 `--tokens` 인자로 공백 분리된 토큰을 직접 입력할 수 있습니다.

출력:
- 단일 입력은 1개 JSON 객체, 다수 입력은 JSON 리스트로 출력합니다.
  각 예측에는 입력 tokens, 예측 labels, attentions(옵션)가 포함됩니다.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from transformers import AutoModelForTokenClassification

from utils import get_tokenizer, setup_cuda, read_json, read_tsv_generic, parse_space_separated_list


def load_labels(path: str) -> List[str]:
    """학습 시 저장한 라벨 목록(labels.json)을 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def predict(tokens: List[str], ckpt_dir: str):
    """토큰 리스트에 대해 IOB 라벨을 예측하고 attention을 함께 반환합니다."""
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
    parser.add_argument("--input_tsv", type=str, default=None, help="\t 구분 TSV 파일 경로 (tokens 열 필요)")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()

    setup_cuda(args.cuda_visible_devices, args.device_index)

    tokens_list: List[List[str]] = []
    if args.input_tsv:
        rows = read_tsv_generic(args.input_tsv)
        for r in rows:
            toks = parse_space_separated_list(r.get("tokens"))
            if toks:
                tokens_list.append(toks)
    elif args.tokens:
        tokens_list.append(args.tokens)

    outputs = [predict(toks, args.ckpt_dir) for toks in tokens_list]
    if len(outputs) == 1:
        print(json.dumps(outputs[0], ensure_ascii=False))
    else:
        print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()


