"""Step 2 - Area A: our own word2vec (trained on the training messages only) + a feed-forward network.

The FFNN is trained with plain mini-batch SGD (+ momentum) and backpropagation, class-weighted
cross-entropy (classes are imbalanced), dropout, and early stopping on validation macro-F1.
"""
import _common
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

from src import config
from src.data import load_split
from src.metrics import save_threshold, scores, tune_threshold
from src.models import FFNN, average_embeddings
from src.normalize import tokenize

_common.seed_everything()
tr, va = load_split("train"), load_split("val")

# ---------------------------------------------------------------- word2vec (skip-gram)
sentences = [tokenize(t) for t in tr.text]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=1, negative=10, epochs=15,
               workers=1, seed=config.SEED)   # workers=1 -> reproducible
w2v.wv.save(str(config.W2V_PATH))
print(f"word2vec: vocabulary={len(w2v.wv):,} words, dim={w2v.wv.vector_size}")
for probe in ("salak", "aptal", "güzel"):
    if probe in w2v.wv:
        print(f"  most similar to {probe!r}:", [w for w, _ in w2v.wv.most_similar(probe, topn=5)])

# ---------------------------------------------------------------- features
Xtr = torch.from_numpy(average_embeddings(w2v.wv, list(tr.text)))
Xva = torch.from_numpy(average_embeddings(w2v.wv, list(va.text)))
ytr = torch.tensor(tr.label.values, dtype=torch.long)

# ---------------------------------------------------------------- FFNN trained with SGD + backprop
net = FFNN(Xtr.shape[1])
net.mu.copy_(Xtr.mean(0))
net.sigma.copy_(Xtr.std(0) + 1e-6)
pos_w = (ytr == 0).sum().item() / max(1, (ytr == 1).sum().item())
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w]))
opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
best_f1, best_state, patience = -1.0, None, 0
for epoch in range(1, 81):
    net.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(net(Xtr[idx]), ytr[idx])
        loss.backward()          # backpropagation
        opt.step()               # SGD update
        total += loss.item() * len(idx)
    net.eval()
    with torch.no_grad():
        pv = torch.softmax(net(Xva), dim=-1)[:, 1].numpy()
    f1 = scores(va.label, pv)["macro_f1"]
    if epoch % 5 == 0 or f1 > best_f1:
        print(f"epoch {epoch:2d}  train loss={total / len(Xtr):.4f}  val macro-F1={f1:.4f}")
    if f1 > best_f1:
        best_f1, best_state, patience = f1, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 10:
            print("early stopping")
            break

net.load_state_dict(best_state)
torch.save(best_state, config.FFNN_PATH)
with torch.no_grad():
    pv = torch.softmax(net(Xva), dim=-1)[:, 1].numpy()
thr = tune_threshold(va.label, pv)
save_threshold("w2v_ffnn", thr)
print(f"-> saved FFNN (best val macro-F1={best_f1:.4f}), threshold={thr:.2f}")
