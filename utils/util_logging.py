from __future__ import annotations

import logging


def configure_logging(verbosity: int = logging.INFO) -> None:
    """프로세스 단위로 간결한 로깅 포맷을 초기화합니다."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=verbosity,
    )


