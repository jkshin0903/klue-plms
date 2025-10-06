"""
KLUE 파인튜닝 스크립트에서 공통으로 사용하는 유틸 함수 모음.

이 모듈은 아래 기능을 제공합니다.
- 재현성을 위한 시드 고정
- 기본 PLM 체크포인트 이름 관리
- 라벨 리스트 직렬화/역직렬화
- 간단한 로깅 설정 및 선택적 EarlyStopping 콜백 생성
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """파이썬, 넘파이, 파이토치의 랜덤 시드를 고정하여 재현성을 확보합니다.

    파이토치는 선택적 의존성이므로 지연 임포트합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_name(default: str = "klue/roberta-large") -> str:
    """기본 PLM 체크포인트 이름을 반환합니다.

    환경변수 MODEL_NAME 이 설정되어 있으면 우선 사용합니다.
    """
    return os.environ.get("MODEL_NAME", default)


def ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성합니다."""
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


def configure_logging(verbosity: int = logging.INFO) -> None:
    """프로세스 단위로 간결한 로깅 포맷을 초기화합니다."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=verbosity,
    )


@dataclass
class EarlyStoppingConfig:
    metric: str = "eval_f1"
    mode: str = "max"  # or "min"
    patience: int = 3


def maybe_add_early_stopping(callbacks: Optional[List] = None, patience: int = 3):
    """EarlyStopping 콜백을 사용할 수 있으면 반환 목록에 추가합니다.

    작성 시점에 transformers 가 없을 수 있으므로, 예외 시 빈 목록을 반환합니다.
    """
    try:
        from transformers import EarlyStoppingCallback

        cb = EarlyStoppingCallback(early_stopping_patience=patience)
        return (callbacks or []) + [cb]
    except Exception:
        return callbacks or []


def setup_cuda(
    cuda_visible_devices: Optional[str] = None,
    device_index: int = 0,
    enable_launch_blocking: bool = True,
    enable_cuda_dsa: bool = True,
) -> None:
    """CUDA 환경 변수를 설정하고 사용할 디바이스를 지정합니다.

    - cuda_visible_devices: 사용 가능한 GPU 리스트를 문자열로 지정 (예: "0" 또는 "0,1").
    - device_index: VISIBLE 목록 내에서 사용할 GPU의 인덱스 (0 기반).
    - enable_launch_blocking: 디버깅 편의를 위한 CUDA_LAUNCH_BLOCKING 설정.
    - enable_cuda_dsa: TORCH_USE_CUDA_DSA 활성화 여부.
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if enable_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if enable_cuda_dsa:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)


