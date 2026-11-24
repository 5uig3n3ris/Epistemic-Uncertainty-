"""Black-box uncertainty baselines: token entropy, semantic entropy, verbalized confidence."""
import re

import numpy as np

CONF_PROMPT = (
    "Question: {q}\nProposed answer: {a}\n"
    "How confident are you that the proposed answer is correct? "
    "Reply with only a number from 0 to 100."
)


def mean_token_entropy(entropies, think_end_idx=None) -> float:
    """Mean entropy over answer tokens (tokens after </think> if a trace exists)."""
    e = entropies[think_end_idx + 1:] if think_end_idx is not None else entropies
    return float(np.mean(e)) if len(e) else float("nan")


def semantic_entropy(labels) -> float:
    """Discrete semantic entropy over cluster labels of sampled answers."""
    counts = np.bincount(np.asarray(labels))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def parse_confidence(text: str):
    m = re.search(r"\d+(?:\.\d+)?", text)
    return min(float(m.group()) / 100.0, 1.0) if m else None


class NLIClusterer:
    """Greedy bidirectional-entailment clustering (Kuhn et al. / Farquhar et al.)."""

    def __init__(self, name: str = "microsoft/deberta-large-mnli", device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval().to(device)
        self.ent = next(i for i, l in self.model.config.id2label.items() if l.lower().startswith("entail"))

    def _entails(self, a: str, b: str) -> bool:
        with self.torch.no_grad():
            x = self.tok(a, b, return_tensors="pt", truncation=True).to(self.model.device)
            return int(self.model(**x).logits.argmax(-1)) == self.ent

    def same(self, q: str, a: str, b: str) -> bool:
        return self._entails(f"{q} {a}", f"{q} {b}") and self._entails(f"{q} {b}", f"{q} {a}")

    def cluster(self, q: str, answers):
        reps, labels = [], []
        for a in answers:
            for i, r in enumerate(reps):
                if self.same(q, r, a):
                    labels.append(i)
                    break
            else:
                reps.append(a)
                labels.append(len(reps) - 1)
        return labels
