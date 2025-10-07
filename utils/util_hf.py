from __future__ import annotations

import json
import os
from typing import List, Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def get_model_name(default: str = "klue/roberta-large") -> str:
    """기본 PLM 체크포인트 이름을 반환합니다.

    환경변수 MODEL_NAME 이 설정되어 있으면 우선 사용합니다.
    """
    return os.environ.get("MODEL_NAME", default)


def read_json(path: str):
    """JSON 파일을 로드하여 파이썬 객체로 반환합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_dataset_dict(train: List[dict], dev: List[dict], test: Optional[List[dict]] = None) -> DatasetDict:
    """list 샘플들을 HuggingFace DatasetDict로 변환합니다.

    test가 비어있으면 포함하지 않습니다.
    """
    dd = {
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(dev),
    }
    if test is not None and len(test) > 0:
        dd["test"] = Dataset.from_list(test)
    return DatasetDict(dd)


def get_tokenizer():
    """기본 모델에 맞는 토크나이저를 생성하여 반환합니다."""
    from .util_hf import get_model_name  # local import to avoid cycles
    return AutoTokenizer.from_pretrained(get_model_name())


