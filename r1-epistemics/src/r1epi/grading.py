"""Answer extraction and correctness labelling (normalized alias match)."""
import re
import string

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_answer(text: str) -> str:
    """Return the text after </think> if present, else the whole text."""
    return text.split("</think>")[-1].strip()


def is_correct(pred: str, golds) -> bool:
    """True if any gold alias equals, or appears as a whole phrase in, the prediction."""
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        g = normalize(g)
        if g and (p == g or re.search(rf"\b{re.escape(g)}\b", p)):
            return True
    return False
