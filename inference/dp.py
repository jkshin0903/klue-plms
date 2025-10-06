"""
KLUE-DP 추론 스크립트: 파인튜닝된 biaffine 파서를 로드하여 헤드/라벨 예측과
attention map을 반환합니다.
"""

from __future__ import annotations

import argparse
import json
from typing import List

import torch
from transformers import AutoTokenizer

from utils import get_model_name, get_tokenizer, setup_cuda, read_json
from finetune.dp import RobertaBiaffineDependencyParser


def load_deprels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def predict(tokens: List[str], ckpt_dir: str):
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
        # Encoder attentions는 finetune.dp의 forward에서 outputs에 담아 반환
        attentions = outputs.get("attentions")

    return {
        "heads": pred_heads,
        "attentions": [a.squeeze(0).tolist() for a in attentions] if attentions is not None else None,
    }


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


