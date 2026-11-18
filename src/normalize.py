"""Area B - regex normalization of noisy Turkish chat text.

Handles: Turkish-aware lower-casing (I->ı, İ->i), URLs / user mentions, leetspeak (4mk -> amk),
letters split by separators (a.m.k / a-m-k / a m k -> amk), elongated letters (salaaaak -> salak),
punctuation/emoji removal, and optional ASCII folding of Turkish characters (ş->s, ğ->g, ...).
"""
import re

_TR_UPPER = str.maketrans({"I": "ı", "İ": "i"})
_ASCII_FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜ", "cgiosuaiuCGIOSU")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
                       "@": "a", "$": "s", "€": "e", "!": "i", "|": "i"})

LETTER = "a-zçğıöşüâîû"
RE_URL = re.compile(r"(https?://\S+|www\.\S+|\bURL\b)", re.I)
RE_USER = re.compile(r"(@\w+)")
CH = rf"[{LETTER}0-9@$€]"
RE_SEPARATED = re.compile(rf"(?<!\S)(?:{CH}[.\-_*+]){{2,}}{CH}(?!\S)")          # a.m.k  s-4-l-a-k
RE_SPACED = re.compile(rf"(?<!\S)(?:{CH} ){{2,}}{CH}(?!\S)")                      # a m k  s 1 k t i r
RE_REPEAT = re.compile(r"(.)\1{2,}")                                               # 3+ repeats -> 1
RE_NONWORD = re.compile(rf"[^{LETTER}0-9<>\s]")
RE_SPACES = re.compile(r"\s+")


def tr_lower(s: str) -> str:
    return s.translate(_TR_UPPER).lower()


def fold_ascii(s: str) -> str:
    return s.translate(_ASCII_FOLD)


def _deleet_token(tok: str) -> str:
    core = tok.strip(".,;:?!'\"()[]{}")                    # keep sentence punctuation out of leetspeak
    if core.startswith("<") or not re.search(f"[{LETTER}]", core):
        return tok                                          # numbers such as 2023 stay numbers
    return tok.replace(core, core.translate(_LEET))


def _join(m: re.Match) -> str:
    """Join obfuscated letter sequences (only if they contain at least two letters)."""
    joined = re.sub(r"[.\-_*+ ]", "", m.group(0))
    if len(re.findall(f"[{LETTER}]", joined, re.I)) < 2:
        return m.group(0)
    return joined.translate(_LEET)


def normalize(text: str) -> str:
    """Full normalization used by the lexicon matcher, tf-idf and word2vec models."""
    s = RE_URL.sub(" <url> ", text)
    s = RE_USER.sub(" <user> ", s)
    s = tr_lower(s)
    s = " ".join(_deleet_token(t) for t in s.split())
    s = RE_SEPARATED.sub(_join, s)
    s = RE_SPACED.sub(_join, s)
    s = RE_REPEAT.sub(r"\1", s)
    s = s.replace("#", " ")
    s = RE_NONWORD.sub(" ", s)
    return RE_SPACES.sub(" ", s).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def normalize_for_bert(text: str) -> str:
    """Light normalization for the cased BERTurk model: keep case and punctuation, only undo noise."""
    s = " ".join(_deleet_token(t) if not t.startswith("@") else t for t in text.split())
    s = RE_SEPARATED.sub(_join, s)
    s = RE_REPEAT.sub(r"\1", s)
    return RE_SPACES.sub(" ", s).strip()


if __name__ == "__main__":
    for t in ["@USER Saaaalak mısın 4mk!!! URL", "a.m.k bu ne", "S 1 K T İ R", "İyi geceler İstanbul 2023 :)"]:
        print(f"{t!r:40} -> {normalize(t)!r}")
