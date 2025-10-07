from __future__ import annotations

import os
from typing import Optional

import torch


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


