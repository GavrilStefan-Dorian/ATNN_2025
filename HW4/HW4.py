import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_missing_label(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and "???" in x:
        return True
    return False


def spearmanr_np(a: np.ndarray, b: np.ndarray) -> float:
    def rankdata(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        uniq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        avg = sums / cnt
        return avg[inv]

    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum())) + 1e-12
    return float((ra * rb).sum() / denom)


def build_story(precontext: str, sentence: str, ending: str, context_mode: str) -> str:
    precontext = (precontext or "").strip()
    sentence = (sentence or "").strip()
    ending = (ending or "").strip()

    if context_mode == "sentence_only":
        return sentence
    if context_mode == "precontext_sentence":
        return (precontext + " " + sentence).strip() if precontext else sentence
    if context_mode == "full":
        parts = [p for p in [precontext, sentence, ending] if p]
        return " ".join(parts).strip()
    raise ValueError(f"Unknown context_mode={context_mode}")


def load_ambistory(path: str, context_mode: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows: List[Dict[str, Any]] = []
    for rid, v in raw.items():
        story = build_story(v.get("precontext", ""), v.get("sentence", ""), v.get("ending", ""), context_mode)

        avg = None if is_missing_label(v.get("average")) else float(v.get("average"))
        choices = None if is_missing_label(v.get("choices")) else v.get("choices")

        dist = None
        if isinstance(choices, list) and len(choices) > 0:
            hist = np.bincount(np.array(choices, dtype=np.int64), minlength=6)[1:6].astype(np.float32)
            dist = hist / (hist.sum() + 1e-12)

        rows.append({
            "id": str(rid),
            "sample_id": str(v.get("sample_id", rid)),
            "homonym": str(v.get("homonym", "")),
            "story": story,
            "definition": str(v.get("judged_meaning", "")),
            "example": str(v.get("example_sentence", "")),
            "avg": avg,
            "dist": dist,  # None for test
        })
    return rows


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    dist: Optional[torch.Tensor]
    avg: Optional[torch.Tensor]
    sample_id: List[str]


class AmbiStoryDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=256, no_definition=False, no_example=False):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.no_definition = no_definition
        self.no_example = no_example

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        story = "Context: " + r["story"]

        sense_parts = []
        if not self.no_definition:
            sense_parts.append(f"Meaning: {r['definition']}")
        if (not self.no_example) and r.get("example", "").strip():
            sense_parts.append(f"Example: {r['example']}")
        sense = " ".join(sense_parts).strip() or " "

        enc = self.tok(
            story,
            sense,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sample_id": r["sample_id"],
        }
        if r.get("dist") is not None:
            item["dist"] = torch.tensor(r["dist"], dtype=torch.float32)
        if r.get("avg") is not None:
            item["avg"] = torch.tensor(r["avg"], dtype=torch.float32)
        return item


def collate_fn(items):
    input_ids = torch.stack([x["input_ids"] for x in items], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in items], dim=0)
    dist = torch.stack([x["dist"] for x in items], dim=0) if "dist" in items[0] else None
    avg = torch.stack([x["avg"] for x in items], dim=0) if "avg" in items[0] else None
    sample_id = [x["sample_id"] for x in items]
    return Batch(input_ids=input_ids, attention_mask=attention_mask, dist=dist, avg=avg, sample_id=sample_id)


class SensePlausibilityModel(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hid = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hid, 5) 

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.drop(cls)
        return self.head(cls)


def expected_rating(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    vals = torch.arange(1, 6, device=probs.device, dtype=probs.dtype)
    return (probs * vals).sum(dim=-1)


def soft_kldiv_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_dist, reduction="batchmean")


def pairwise_rank_loss(scores: torch.Tensor, gold_avg: torch.Tensor, sample_ids: List[str], margin: float = 0.1) -> torch.Tensor:
    B = scores.size(0)
    loss = torch.zeros((), device=scores.device)
    pairs = 0
    m = torch.tensor(margin, device=scores.device)

    for i in range(B):
        gi = float(gold_avg[i].item())
        for j in range(i + 1, B):
            if sample_ids[i] != sample_ids[j]:
                continue
            gj = float(gold_avg[j].item())
            if gi == gj:
                continue
            y = 1.0 if gi > gj else -1.0
            loss = loss + torch.relu(m - y * (scores[i] - scores[j]))
            pairs += 1

    if pairs == 0:
        return torch.zeros((), device=scores.device)
    return loss / pairs


@torch.no_grad()
def eval_spearman(model, loader, device) -> float:
    model.eval()
    preds, golds = [], []
    for batch in loader:
        logits = model(batch.input_ids.to(device), batch.attention_mask.to(device))
        exp = expected_rating(logits).detach().cpu().numpy()
        if batch.avg is None:
            continue
        g = batch.avg.detach().cpu().numpy()
        preds.append(exp)
        golds.append(g)
    if not preds:
        return float("nan")
    p = np.concatenate(preds)
    g = np.concatenate(golds)
    return spearmanr_np(p, g)


@torch.no_grad()
def predict_int(model, loader, device) -> List[int]:
    model.eval()
    out: List[int] = []
    for batch in loader:
        logits = model(batch.input_ids.to(device), batch.attention_mask.to(device))
        exp = expected_rating(logits)
        ints = torch.clamp(torch.round(exp), 1, 5).long().detach().cpu().tolist()
        out.extend(ints)
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]], preds: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": str(r["id"]), "prediction": int(p)}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/best")
    ap.add_argument("--base_model", type=str, default="microsoft/deberta-v3-large")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", choices=["full", "precontext_sentence", "sentence_only"], default="full")
    ap.add_argument("--no_definition", action="store_true")
    ap.add_argument("--no_example", action="store_true")
    ap.add_argument("--rank_lambda", type=float, default=0.2)
    ap.add_argument("--rank_margin", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    wb_run = None
    if args.wandb:
        import wandb  

        wandb_key = os.getenv("WANDB_API_KEY", "").strip()
        if wandb_key:
            wandb.login(key=wandb_key)

        wb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "semeval2026-task5"),
            name=os.getenv("WANDB_RUN_NAME", None),
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    train_rows = load_ambistory(args.train, args.context)
    dev_rows = load_ambistory(args.dev, args.context)
    test_rows = load_ambistory(args.test, args.context)

    train_ds = AmbiStoryDataset(train_rows, tok, max_len=args.max_len, no_definition=args.no_definition, no_example=args.no_example)
    dev_ds = AmbiStoryDataset(dev_rows, tok, max_len=args.max_len, no_definition=args.no_definition, no_example=args.no_example)
    test_ds = AmbiStoryDataset(test_rows, tok, max_len=args.max_len, no_definition=args.no_definition, no_example=args.no_example)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,             
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = SensePlausibilityModel(args.base_model).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grouped, lr=args.lr)

    total_steps = args.epochs * max(1, math.ceil(len(train_loader) / args.grad_accum))
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    best_sp = -1e9
    best_path = os.path.join(args.out_dir, "best.pt")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        steps = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for it, batch in enumerate(train_loader, start=1):
            x = batch.input_ids.to(device, non_blocking=True)
            m = batch.attention_mask.to(device, non_blocking=True)
            target_dist = batch.dist.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                logits = model(x, m)
                loss_main = soft_kldiv_loss(logits, target_dist)

                loss_rank = torch.zeros((), device=device)
                if args.rank_lambda > 0.0 and batch.avg is not None:
                    scores = expected_rating(logits)
                    gold = batch.avg.to(device, non_blocking=True)
                    loss_rank = pairwise_rank_loss(scores, gold, batch.sample_id, margin=args.rank_margin)

                loss = (loss_main + args.rank_lambda * loss_rank) / args.grad_accum

            scaler.scale(loss).backward()

            if it % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            running += float(loss.item())
            steps += 1

        sp = eval_spearman(model, dev_loader, device)
        print(f"Epoch {epoch}: train_loss={running/max(1,steps):.4f} dev_spearman={sp:.4f}")

        if args.wandb:
            import wandb
            gpu_mem_gb = 0.0
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            wandb.log({
                "epoch": epoch,
                "train_loss": running / max(1, steps),
                "dev_spearman_cont": sp,
                "gpu_mem_peak_gb": gpu_mem_gb,
            })

        if sp > best_sp:
            best_sp = sp
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    print(f"Best dev Spearman: {best_sp:.4f}  Time: {time.time()-t0:.1f}s")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    dev_pred = predict_int(model, dev_loader, device)
    test_pred = predict_int(model, test_loader, device)

    dev_out = os.path.join(args.out_dir, "dev_predictions.jsonl")
    test_out = os.path.join(args.out_dir, "predictions.jsonl")

    write_jsonl(dev_out, dev_rows, dev_pred)
    write_jsonl(test_out, test_rows, test_pred)

    print("Wrote:", dev_out)
    print("Wrote:", test_out)

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
