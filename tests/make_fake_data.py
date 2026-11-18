"""Creates SYNTHETIC files in the OffensEval-TR raw format + a tiny random BERT, for the smoke test only."""
import os
import random
from pathlib import Path

random.seed(0)
data = Path(os.environ["DATA_DIR"]) / "raw" / "offenseval2020-turkish"
(data / "offenseval-tr-training-v1").mkdir(parents=True, exist_ok=True)
(data / "offenseval-tr-testset-v1").mkdir(parents=True, exist_ok=True)

clean = "maç güzel bugün hava takım oyun gol hakem sezon kazandı iyi akşamlar teşekkürler harika film".split()
bad = ["salak", "aptal", "gerizekalı", "mal", "ahmak"]
noisy = {"salak": ["s4l4k", "saaalak", "s.a.l.a.k"], "aptal": ["4ptal", "aptaaal"], "mal": ["m4l"]}


def sentence(off):
    words = random.sample(clean, 5)
    if off:
        w = random.choice(bad)
        w = random.choice(noisy.get(w, [w]) + [w, w])
        words.insert(random.randrange(5), w)
    if random.random() < .3:
        words.insert(0, "@USER")
    return " ".join(words)


with open(data / "offenseval-tr-training-v1" / "offenseval-tr-training-v1.tsv", "w") as f:
    f.write("id\ttweet\tsubtask_a\n")
    for i in range(2000):
        off = random.random() < .2
        f.write(f"{i}\t{sentence(off)}\t{'OFF' if off else 'NOT'}\n")
with open(data / "offenseval-tr-testset-v1" / "offenseval-tr-testset-v1.tsv", "w") as f, \
        open(data / "offenseval-tr-testset-v1" / "offenseval-tr-labela-v1.tsv", "w") as g:
    f.write("id\ttweet\n")
    for i in range(5000, 5300):
        off = random.random() < .2
        f.write(f"{i}\t{sentence(off)}\n")
        g.write(f"{i},{'OFF' if off else 'NOT'}\n")

# tiny BERT
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer  # noqa: E402

out = Path(os.environ["BERT_MODEL"])
out.mkdir(parents=True, exist_ok=True)
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(clean + bad + list("abcdefghijklmnoprstuvyzçğıöşü0123456789@.!")))
(out / "vocab.txt").write_text("\n".join(vocab))
BertTokenizer(str(out / "vocab.txt"), do_lower_case=False).save_pretrained(out)
cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
BertForSequenceClassification(cfg).save_pretrained(out)
print("fake data + tiny BERT written")
