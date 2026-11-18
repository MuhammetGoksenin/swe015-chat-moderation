"""Step 1 - classical baselines (Area B).

 a) Lexicon + minimum-edit-distance matcher (rule baseline). Lexicon size and edit-distance threshold
    are chosen on the validation split.
 b) TF-IDF (word 1-2 grams + char 2-5 grams on normalized text) + logistic regression.
"""
import _common
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from src import config
from src.data import load_split
from src.lexicon import LexiconMatcher, build_lexicon
from src.metrics import save_threshold, scores, tune_threshold
from src.normalize import normalize

_common.seed_everything()
tr, va = load_split("train"), load_split("val")

# ---------------------------------------------------------------- a) lexicon rule baseline
best = None
for top_k in (100, 200, 300, 500):
    words = build_lexicon(tr.text, tr.label, top_k=top_k)
    for max_dist in (0, 1):
        m = LexiconMatcher(words, max_dist=max_dist)
        f1 = f1_score(va.label, (np.asarray(m.predict_proba(va.text)) >= 0.5).astype(int), average="macro")
        print(f"lexicon top_k={top_k:3d} max_dist={max_dist}: val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, m)
best[1].save(config.LEXICON_PATH)
save_threshold("lexicon", 0.5)
print(f"-> saved lexicon with {len(best[1].words)} words, max_dist={best[1].max_dist}")
print("   top entries:", best[1].words[:15])

# ---------------------------------------------------------------- b) tf-idf + logistic regression
best = None
for C in (1.0, 4.0, 10.0):
    pipe = Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     sublinear_tf=True, max_features=200_000)),
        ])),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(tr.text, tr.label)
    pv = pipe.predict_proba(va.text)[:, 1]
    thr = tune_threshold(va.label, pv)
    s = scores(va.label, pv, thr)
    print(f"tfidf C={C:4.1f}: val macro-F1={s['macro_f1']:.4f} (thr={thr:.2f})")
    if best is None or s["macro_f1"] > best[0]:
        best = (s["macro_f1"], pipe, thr)
joblib.dump(best[1], config.TFIDF_PATH, compress=3)
save_threshold("tfidf", best[2])
print(f"-> saved tf-idf model, threshold={best[2]:.2f}")
