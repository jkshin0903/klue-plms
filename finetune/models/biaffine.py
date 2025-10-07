from __future__ import annotations

import torch
import torch.nn as nn


class Biaffine(nn.Module):
    """biaffine 점수화 모듈.

    두 입력 벡터 x(의존어), y(지배어)에 대해 bilinear + linear 합으로 점수를 계산합니다.
    arc(head 선택)과 relation(라벨) 모두에 공통으로 사용됩니다.
    """

    def __init__(self, in1: int, in2: int, out: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(out, in1, in2))
        self.W1 = nn.Linear(in1, out, bias=False)
        self.W2 = nn.Linear(in2, out, bias=True)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bilinear = torch.einsum(
            "blo,ohlm->blm", torch.einsum("blh,ohl->blo", x, self.U.sum(dim=2)), y.transpose(1, 2)
        )
        w1 = self.W1(x)
        w2 = self.W2(y)
        scores = bilinear.unsqueeze(-1) + w1.unsqueeze(2) + w2.unsqueeze(1)
        return scores.squeeze(-1)


