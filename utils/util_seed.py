from __future__ import annotations

import random

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


