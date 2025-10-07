from __future__ import annotations

import torch
import torch.nn as nn

from .roberta_biaffine_dp import RobertaBiaffineDependencyParser


class Wrapper(nn.Module):
    """Trainer 호환 래퍼.

    - loss, arc_scores만 노출하여 평가 시 head 정확도를 계산합니다.
    """

    def __init__(self, base: RobertaBiaffineDependencyParser):
        super().__init__()
        self.base = base

    def forward(self, **batch):
        out = self.base(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            heads=batch.get("heads"),
            deprels=batch.get("deprels"),
            word_starts=batch.get("word_starts"),
        )
        return {"loss": out.get("loss", None), "arc_scores": out["arc_scores"]}


