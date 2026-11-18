"""Step 3 - Area C: fine-tune BERTurk (dbmdz/bert-base-turkish-cased) with a classification head.

Plain PyTorch loop (no Trainer -> fewer version issues): AdamW, linear warm-up/decay, mixed precision on
GPU, class-weighted loss, best epoch selected on validation macro-F1. Run on a Colab GPU (T4 is enough).
"""
import _common
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config
from src.data import load_split
from src.metrics import save_threshold, scores, tune_threshold
from src.normalize import normalize_for_bert

_common.seed_everything()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| base model:", config.BERT_MODEL)

tr, va = load_split("train"), load_split("val")
tok = AutoTokenizer.from_pretrained(config.BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    config.BERT_MODEL, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}).to(device)


def make_loader(df, shuffle):
    items = list(zip([normalize_for_bert(t) for t in df.text], df.label.tolist()))

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=config.BERT_MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return DataLoader(items, batch_size=config.BERT_BATCH, shuffle=shuffle, collate_fn=collate)


train_dl, val_dl = make_loader(tr, True), make_loader(va, False)
pos_w = (tr.label == 0).sum() / max(1, (tr.label == 1).sum())
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(np.sqrt(pos_w))], device=device))
opt = torch.optim.AdamW(model.parameters(), lr=config.BERT_LR, weight_decay=0.01)
steps = len(train_dl) * config.BERT_EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def predict(dl):
    model.eval()
    out = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type=device, enabled=use_amp):
            out.append(torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


best_f1 = -1.0
for epoch in range(1, config.BERT_EPOCHS + 1):
    model.train()
    total = 0.0
    for step, b in enumerate(train_dl, 1):
        b = {k: v.to(device) for k, v in b.items()}
        labels = b.pop("labels")
        with torch.autocast(device_type=device, enabled=use_amp):
            loss = loss_fn(model(**b).logits.float(), labels)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
        if step % 200 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_dl)} loss={total / step:.4f}")
    pv = predict(val_dl)
    f1 = scores(va.label, pv)["macro_f1"]
    print(f"epoch {epoch}: train loss={total / len(train_dl):.4f}  val macro-F1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained(config.BERT_FP32_DIR, safe_serialization=True)
        tok.save_pretrained(config.BERT_FP32_DIR)
        thr = tune_threshold(va.label, pv)
        save_threshold("bert_fp32", thr)
        print(f"  -> saved best model to {config.BERT_FP32_DIR} (threshold={thr:.2f})")

print(f"best val macro-F1={best_f1:.4f}")
