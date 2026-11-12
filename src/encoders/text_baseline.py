"""
DeepFusionGuard - text baseline (RoBERTa on HC3: human vs ChatGPT)

Splits are question-disjoint (same idea as identity-disjoint splits for faces):
a question and ALL its answers go to exactly one of train / val / test.

Usage (Colab T4):
  python text_baseline.py --data all.jsonl --out outputs/results
Quick test (2 minutes):
  python text_baseline.py --data all.jsonl --out outputs/results --max_train 2000 --epochs 1
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 42


def load_hc3(path):
    """Return list of (question_id, text, label); label 0 = human, 1 = ChatGPT."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for qid, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            for t in d.get("human_answers", []):
                if isinstance(t, str) and len(t.strip()) >= 20:
                    rows.append((qid, t.strip(), 0))
            for t in d.get("chatgpt_answers", []):
                if isinstance(t, str) and len(t.strip()) >= 20:
                    rows.append((qid, t.strip(), 1))
    return rows


def split_by_question(rows, seed=SEED):
    qids = sorted({r[0] for r in rows})
    random.Random(seed).shuffle(qids)
    n = len(qids)
    train_q = set(qids[: int(0.70 * n)])
    val_q = set(qids[int(0.70 * n): int(0.85 * n)])
    parts = {"train": [], "val": [], "test": []}
    for r in rows:
        key = "train" if r[0] in train_q else "val" if r[0] in val_q else "test"
        parts[key].append(r)
    return parts


def balance(rows, cap=0, seed=SEED):
    """Equal number of human / ChatGPT samples; optional cap on total size."""
    rng = random.Random(seed)
    h = [r for r in rows if r[2] == 0]
    c = [r for r in rows if r[2] == 1]
    k = min(len(h), len(c))
    if cap:
        k = min(k, cap // 2)
    out = rng.sample(h, k) + rng.sample(c, k)
    rng.shuffle(out)
    return out


def make_loader(rows, tok, max_len, bs, shuffle):
    def collate(batch):
        enc = tok([b[1] for b in batch], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[2] for b in batch])
        return enc
    return DataLoader(rows, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(**batch).logits
        probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        labels.extend(y.cpu().tolist())
    return np.array(labels), np.array(probs)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": pr, "recall": rc,
            "f1": f1, "auc": roc_auc_score(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="HC3 .jsonl file (e.g. all.jsonl)")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_train", type=int, default=20000, help="cap on training samples (0 = no cap)")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = load_hc3(args.data)
    parts = split_by_question(rows)
    train = balance(parts["train"], cap=args.max_train)
    val = balance(parts["val"], cap=4000)
    test = balance(parts["test"], cap=0)
    print(f"samples  train={len(train)}  val={len(val)}  test={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    tr_dl = make_loader(train, tok, args.max_len, args.bs, True)
    va_dl = make_loader(val, tok, args.max_len, args.bs * 2, False)
    te_dl = make_loader(test, tok, args.max_len, args.bs * 2, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_f1, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for i, batch in enumerate(tr_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
            if i % 50 == 0:
                print(f"epoch {ep + 1}  step {i}/{len(tr_dl)}  loss {total / i:.4f}")
        yv, pv = predict(model, va_dl, device)
        m = metrics(yv, pv)
        print(f"epoch {ep + 1} val:", {k: round(v, 4) for k, v in m.items()})
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    yt, pt = predict(model, te_dl, device)
    res = metrics(yt, pt)
    print("\nTEST:", {k: round(v, 4) for k, v in res.items()})

    (out / "text_baseline_metrics.json").write_text(json.dumps(
        {"model": args.model, "n_train": len(train), "n_val": len(val), "n_test": len(test),
         "test": res}, indent=2))

    cm = confusion_matrix(yt, (pt >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "ChatGPT"]); ax.set_yticklabels(["Human", "ChatGPT"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("HC3 text baseline")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out / "text_confusion_matrix.png", dpi=200)
    print("saved results to", out)


if __name__ == "__main__":
    main()
