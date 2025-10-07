"""
KLUE-DP 추론 스크립트

개요:
- 파인튜닝된 biaffine 의존 구문 분석기를 로드하여 입력 토큰에 대한 head 예측을 수행합니다.
- 학습 시 모델에서 노출한 encoder attention map도 함께 반환할 수 있습니다.

입력/출력:
- 입력: 공백 분리된 토큰 리스트(`--tokens`), 체크포인트 디렉터리(`--ckpt_dir`).
- 출력 JSON:
  {
    "heads": [int, ...],  # 각 토큰 위치의 예측 head 인덱스(워드피스 기준 argmax 결과)
    "attentions": [[[...]] or null]  # 레이어별 attention(옵션)
  }
"""

from __future__ import annotations

import argparse
import json
from typing import List

import torch
from transformers import AutoTokenizer

from utils import get_model_name, get_tokenizer, setup_cuda, read_json, read_tsv_generic, parse_space_separated_list
from finetune.dp import RobertaBiaffineDependencyParser


def load_deprels(path: str) -> List[str]:
    """학습 시 저장한 의존관계 라벨 목록을 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def predict(tokens: List[str], ckpt_dir: str):
    """토큰 리스트에 대해 head 인덱스를 예측합니다.

    - 학습 시 저장된 `deprels.json`을 로드해 라벨 공간 크기를 복원합니다.
    - 파라미터는 `pytorch_model.bin`에서 state_dict 형태로 복원합니다.
    """
    tokenizer = get_tokenizer()
    deprels = load_deprels(f"{ckpt_dir}/deprels.json")

    model = RobertaBiaffineDependencyParser(get_model_name(), num_deprel=len(deprels))
    state = torch.load(f"{ckpt_dir}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            heads=None,
            deprels=None,
            word_starts=None,
        )
        arc_scores = outputs["arc_scores"]  # [1,L,L]
        pred_heads = arc_scores.argmax(-1).squeeze(0).tolist()
        # Encoder attentions는 finetune.dp의 forward에서 outputs에 담아 반환됩니다.
        attentions = outputs.get("attentions")

    return {
        "heads": pred_heads,
        "attentions": [a.squeeze(0).tolist() for a in attentions] if attentions is not None else None,
    }


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


