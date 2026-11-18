"""Area B - rule baseline: an offensive-word lexicon matched with minimum edit distance.

The lexicon is NOT hand-written: it is extracted from the TRAINING split only, as the tokens with the
highest smoothed log-odds ratio of appearing in OFF vs. NOT messages. A message is flagged when one of
its (normalized, ASCII-folded) tokens - or the stem-sized prefix of a token, to cover Turkish suffixes
(salaklar, salaksın) - is within a small Levenshtein distance of a lexicon entry (covers typos such as
"slak", "aptl").
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

from .normalize import fold_ascii, normalize

IGNORE = {"<user>", "<url>"}


def edit_distance(a: str, b: str, max_d: int = 99) -> int:
    """Classic Levenshtein DP (insert/delete/substitute = 1) with an early-exit band."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


def _tokens(text: str) -> list[str]:
    return [fold_ascii(t) for t in normalize(text).split() if t not in IGNORE]


def build_lexicon(texts, labels, top_k: int = 300, min_count: int = 5, min_len: int = 3) -> list[str]:
    off, clean = Counter(), Counter()
    for t, y in zip(texts, labels):
        (off if y == 1 else clean).update(set(_tokens(t)))
    n_off, n_clean = sum(off.values()), sum(clean.values())
    scores = {}
    for w in set(off) | set(clean):
        if len(w) < min_len or off[w] + clean[w] < min_count or w.isdigit():
            continue
        p_off = (off[w] + 1) / (n_off + 2)
        p_clean = (clean[w] + 1) / (n_clean + 2)
        scores[w] = math.log(p_off) - math.log(p_clean)
    return [w for w, s in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k] if s > 0]


class LexiconMatcher:
    def __init__(self, words, max_dist: int = 1):
        self.words = sorted(set(words))
        self.max_dist = max_dist
        self.exact = set(self.words)
        self._cache: dict[str, str | None] = {}

    def _thr(self, w: str) -> int:
        return 0 if len(w) <= 3 else (self.max_dist if len(w) <= 6 else self.max_dist + 1)

    def match_token(self, tok: str):
        if tok not in self._cache:
            self._cache[tok] = self._match(tok)
        return self._cache[tok]

    def _match(self, tok: str):
        if tok in self.exact:
            return tok
        for w in self.words:
            if tok[0] != w[0] and tok[:2] != w[1:3] and tok[1:3] != w[:2]:
                continue                                  # cheap filter: some overlap at the start
            thr = self._thr(w)
            if thr == 0:
                if tok.startswith(w) and len(w) >= 3 and len(tok) - len(w) <= 4:
                    return w
                continue
            # whole token (typos) or stem-length prefix (Turkish suffixes)
            if edit_distance(tok, w, thr) <= thr or (len(tok) > len(w) and edit_distance(tok[: len(w)], w, thr) <= thr):
                return w
        return None

    def matches(self, text: str) -> list[str]:
        return [m for m in (self.match_token(t) for t in _tokens(text)) if m]

    def predict_proba(self, texts) -> list[float]:
        """Rule 'confidence': 0.1 with no hit, 0.75 for one hit, rising with more hits."""
        out = []
        for t in texts:
            n = len(self.matches(t))
            out.append(0.1 if n == 0 else min(0.99, 0.75 + 0.1 * (n - 1)))
        return out

    def mask(self, text: str) -> tuple[str, int]:
        """Replace offensive words in the ORIGINAL text with asterisks. Returns (masked, n_masked)."""
        out, n = [], 0
        for raw in re.split(r"(\s+)", text):
            toks = _tokens(raw) if raw.strip() else []
            if toks and any(self.match_token(t) for t in toks):
                out.append(raw[0] + "*" * (len(raw) - 1) if len(raw) > 1 else "*")
                n += 1
            else:
                out.append(raw)
        return "".join(out), n

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"max_dist": self.max_dist, "words": self.words}, ensure_ascii=False, indent=0))

    @classmethod
    def load(cls, path: Path) -> "LexiconMatcher":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["words"], d["max_dist"])
