from __future__ import annotations

import json
import os
from typing import List, Sequence


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_label_list(labels: Sequence[str], path: str) -> None:
    """라벨 리스트를 JSON으로 저장합니다 (인덱스 == 라벨 ID)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(labels), f, ensure_ascii=False, indent=2)


def load_label_list(path: str) -> List[str]:
    """JSON에서 라벨 리스트를 로드합니다 (인덱스 == 라벨 ID)."""
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


