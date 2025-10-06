"""
klue/roberta-large 기반 KLUE-DP(의존 구문 분석) 파인튜닝 스크립트.
"""

from __future__ import annotations
import logging
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from utils import (
    configure_logging,
    get_model_name,
    save_label_list,
    set_seed,
    setup_cuda,
    read_json,
    make_dataset_dict,
    get_tokenizer,
)


def load_klue_dp(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    def pick(samples: List[Dict]) -> List[Dict]:
        out = []
        for s in samples:
            out.append(
                {
                    "tokens": s.get("tokens", []),
                    "heads": s.get("heads", []),  # 1-based head, 0 for ROOT
                    "deprels": s.get("deprels", []),
                }
            )
        return out

    train = pick(read_json(os.path.join(data_dir, "train.json")))
    dev = pick(read_json(os.path.join(data_dir, "dev.json")))
    test_path = os.path.join(data_dir, "test.json")
    test = pick(read_json(test_path)) if os.path.exists(test_path) else []
    return train, dev, test


def build_deprel_list(datasets: Tuple[List[Dict], List[Dict], List[Dict]]) -> List[str]:
    labels = set(["root"])  # 기본값 포함
    for split in datasets:
        for s in split:
            labels.update(s.get("deprels", []))
    return sorted(labels)


def tokenize_and_align_for_dp(examples, tokenizer, deprel_to_id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        return_offsets_mapping=False,
    )

    batch_heads = []
    batch_deprels = []
    batch_word_starts = []

    for i, (heads, deprels) in enumerate(zip(examples["heads"], examples["deprels"])):
        word_ids = tokenized.word_ids(batch_index=i)
        word_to_first_wp = {}
        seen = set()
        for idx, w in enumerate(word_ids):
            if w is None:
                continue
            if w not in seen:
                word_to_first_wp[w] = idx
                seen.add(w)

        num_words = max([-1] + [w for w in word_to_first_wp.keys()]) + 1 if word_to_first_wp else 0
        if num_words != len(heads) or num_words != len(deprels):
            cut = min(num_words, len(heads), len(deprels))
            heads = heads[:cut]
            deprels = deprels[:cut]

        word_starts = [0] * len(word_ids)
        for w, wp_idx in word_to_first_wp.items():
            word_starts[wp_idx] = 1

        mapped_heads = []
        mapped_deprels = []
        for dep_word_idx, (h, rel) in enumerate(zip(heads, deprels)):
            if dep_word_idx not in word_to_first_wp:
                continue
            dep_wp = word_to_first_wp[dep_word_idx]
            if h == 0:
                head_wp = -1
            else:
                hw = h - 1
                head_wp = word_to_first_wp.get(hw, -1)
            mapped_heads.append(head_wp)
            mapped_deprels.append(deprel_to_id.get(rel, deprel_to_id["root"]))

        batch_heads.append(mapped_heads)
        batch_deprels.append(mapped_deprels)
        batch_word_starts.append(word_starts)

    tokenized["heads"] = batch_heads
    tokenized["deprels"] = batch_deprels
    tokenized["word_starts"] = batch_word_starts
    return tokenized


class Biaffine(nn.Module):
    def __init__(self, in1: int, in2: int, out: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(out, in1, in2))
        self.W1 = nn.Linear(in1, out, bias=False)
        self.W2 = nn.Linear(in2, out, bias=True)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bilinear = torch.einsum("blo,ohlm->blm", torch.einsum("blh,ohl->blo", x, self.U.sum(dim=2)), y.transpose(1, 2))
        w1 = self.W1(x)
        w2 = self.W2(y)
        scores = bilinear.unsqueeze(-1) + w1.unsqueeze(2) + w2.unsqueeze(1)
        return scores.squeeze(-1)


class RobertaBiaffineDependencyParser(nn.Module):
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

        mask = word_starts if word_starts is not None else attention_mask
        dep_h = self.dep_mlp(enc)
        head_h = self.head_mlp(enc)
        arc_scores = self.arc_biaffine(dep_h, head_h)
        minus_inf = -1e4
        head_mask = (mask > 0).unsqueeze(1).expand(-1, arc_scores.size(1), -1)
        arc_scores = arc_scores.masked_fill(~head_mask, minus_inf)

        outputs = {"arc_scores": arc_scores, "attentions": attentions}

        if heads is not None and deprels is not None:
            B, L = input_ids.size()
            device = input_ids.device
            valid_positions = (mask > 0)
            arc_logit = arc_scores
            target_heads = torch.full((B, L), fill_value=-100, dtype=torch.long, device=device)
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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune klue/roberta-large on KLUE-DP")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="예: '0' 또는 '0,1'")
    parser.add_argument("--device_index", type=int, default=0, help="VISIBLE 목록 내 사용 GPU 인덱스")
    parser.add_argument("--no_launch_blocking", action="store_true")
    parser.add_argument("--no_cuda_dsa", action="store_true")
    args = parser.parse_args()

    configure_logging()
    setup_cuda(
        cuda_visible_devices=args.cuda_visible_devices,
        device_index=args.device_index,
        enable_launch_blocking=not args.no_launch_blocking,
        enable_cuda_dsa=not args.no_cuda_dsa,
    )
    set_seed(args.seed)

    train, dev, test = load_klue_dp(args.data_dir)
    deprel_list = build_deprel_list((train, dev, test))
    deprel_to_id = {l: i for i, l in enumerate(deprel_list)}

    ds = make_dataset_dict(train, dev, test)

    tokenizer = get_tokenizer()
    tokenized = ds.map(
        lambda ex: tokenize_and_align_for_dp(ex, tokenizer, deprel_to_id),
        batched=True,
        remove_columns=["tokens", "heads", "deprels"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_label_list(deprel_list, os.path.join(args.output_dir, "deprels.json"))

    model = RobertaBiaffineDependencyParser(get_model_name(), num_deprel=len(deprel_list))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="head_accuracy",
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        report_to=["none"],
    )

    class Wrapper(nn.Module):
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

    wrapped = Wrapper(model)

    def collate_fn(features: List[Dict]):
        batch = default_data_collator(
            [
                {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
                for f in features
            ]
        )
        def pad_list(list_of_lists, pad=-100):
            maxlen = max((len(x) for x in list_of_lists), default=0)
            return [x + [pad] * (maxlen - len(x)) for x in list_of_lists]

        batch["heads"] = torch.tensor(pad_list([f["heads"] for f in features]), dtype=torch.long)
        batch["deprels"] = torch.tensor(pad_list([f["deprels"] for f in features]), dtype=torch.long)
        batch["word_starts"] = torch.tensor(
            pad_list([f["word_starts"] for f in features], pad=0), dtype=torch.long
        )
        return batch

    def compute_metrics_eval(eval_pred):
        arc_scores = eval_pred.predictions
        label_heads = eval_pred.label_ids
        pred_heads = arc_scores.argmax(-1)
        mask = label_heads != -100
        acc = (pred_heads[mask] == label_heads[mask]).mean() if mask.any() else 0.0
        return {"head_accuracy": float(acc)}

    trainer = Trainer(
        model=wrapped,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_eval,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logging.getLogger(__name__).info("Eval metrics: %s", metrics)

    if "test" in tokenized:
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])  # type: ignore
        logging.getLogger(__name__).info("Test metrics: %s", test_metrics)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    main()


