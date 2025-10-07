from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .biaffine import Biaffine


class RobertaBiaffineDependencyParser(nn.Module):
    """RoBERTa 인코더 + biaffine 기반 의존 구문 분석기.

    - dep/head MLP로 인코더 은닉을 저차원 공간으로 투영
    - arc_biaffine으로 head 선택 점수 계산, rel_biaffine으로 관계 라벨 점수 계산
    - 학습 시 arc(전체 토큰에 대한 다중분류)와 관계(유효 head 쌍에 대한 다중분류) 손실을 합산
    """

    def __init__(self, encoder_name: str, num_deprel: int, hidden_mlp: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size
        self.dep_mlp = nn.Sequential(nn.Linear(enc_dim, hidden_mlp), nn.ReLU())
        self.head_mlp = nn.Sequential(nn.Linear(enc_dim, hidden_mlp), nn.ReLU())
        self.arc_biaffine = Biaffine(hidden_mlp, hidden_mlp, out=1)
        self.rel_biaffine = Biaffine(hidden_mlp, hidden_mlp, out=num_deprel)

    def forward(self, input_ids, attention_mask, heads=None, deprels=None, word_starts=None):
        enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        enc = enc_outputs.last_hidden_state
        attentions = enc_outputs.attentions  # Tuple[Layer] of [B,H,L,L]

        # 첫 서브워드 위치만을 유효 토큰으로 간주하여 arc/rel 계산 범위를 제한합니다.
        mask = word_starts if word_starts is not None else attention_mask
        dep_h = self.dep_mlp(enc)
        head_h = self.head_mlp(enc)
        arc_scores = self.arc_biaffine(dep_h, head_h)
        minus_inf = -1e4  # 마스킹을 위한 큰 음수
        head_mask = (mask > 0).unsqueeze(1).expand(-1, arc_scores.size(1), -1)
        arc_scores = arc_scores.masked_fill(~head_mask, minus_inf)

        outputs = {"arc_scores": arc_scores, "attentions": attentions}

        if heads is not None and deprels is not None:
            B, L = input_ids.size()
            device = input_ids.device
            valid_positions = (mask > 0)  # 첫 서브워드 위치
            arc_logit = arc_scores
            target_heads = torch.full((B, L), fill_value=-100, dtype=torch.long, device=device)
            # 단어 첫 서브워드 위치에 대해 gold head(워드피스 인덱스)를 채웁니다.
            for b in range(B):
                idxs = torch.nonzero(valid_positions[b], as_tuple=False).squeeze(-1).tolist()
                if not idxs:
                    continue
                gold_heads = heads[b][: len(idxs)]
                for j, dep_wp in enumerate(idxs):
                    gh = gold_heads[j] if j < len(gold_heads) else -1
                    if gh >= 0 and gh < L:
                        target_heads[b, dep_wp] = gh
                    else:
                        target_heads[b, dep_wp] = -100

            arc_loss = F.cross_entropy(arc_logit.transpose(1, 2), target_heads, ignore_index=-100)

            rel_logits_list = []
            rel_targets = []
            # 관계 라벨은 (유효 dep, 유효 head) 쌍에 대해서만 계산합니다.
            for b in range(B):
                idxs = torch.nonzero(valid_positions[b], as_tuple=False).squeeze(-1).tolist()
                if not idxs:
                    continue
                gold_heads = heads[b][: len(idxs)]
                gold_rels = deprels[b][: len(idxs)]
                for j, dep_wp in enumerate(idxs):
                    gh = gold_heads[j] if j < len(gold_heads) else -1
                    if gh < 0 or gh >= L:
                        continue
                    dep_vec = dep_h[b, dep_wp : dep_wp + 1]
                    head_vec = head_h[b, gh : gh + 1]
                    rel_scores = self.rel_biaffine(dep_vec, head_vec).squeeze(0).squeeze(0)
                    rel_logits_list.append(rel_scores)
                    rel_targets.append(gold_rels[j])

            if rel_logits_list:
                rel_logits = torch.stack(rel_logits_list, dim=0)
                rel_targets_t = torch.tensor(rel_targets, dtype=torch.long, device=device)
                rel_loss = F.cross_entropy(rel_logits, rel_targets_t)
            else:
                rel_loss = torch.tensor(0.0, device=device)

            outputs["loss"] = arc_loss + rel_loss

        return outputs


