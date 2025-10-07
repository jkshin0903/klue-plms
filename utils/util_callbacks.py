from __future__ import annotations

from typing import List, Optional


def maybe_add_early_stopping(callbacks: Optional[List] = None, patience: int = 3):
    """EarlyStopping 콜백을 사용할 수 있으면 반환 목록에 추가합니다.

    transformers가 설치되어 있지 않은 환경에서도 안전하도록 예외 시 원본 콜백 목록을 반환합니다.
    """
    try:
        from transformers import EarlyStoppingCallback

        cb = EarlyStoppingCallback(early_stopping_patience=patience)
        return (callbacks or []) + [cb]
    except Exception:
        return callbacks or []


