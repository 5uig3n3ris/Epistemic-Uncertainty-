"""Black-box baselines: token entropy, semantic entropy, verbalized confidence."""
import argparse
import json
import pathlib

import numpy as np
from tqdm import tqdm

from r1epi import model as M
from r1epi.calibration import auroc_error_detection, ece
from r1epi.uncertainty import (CONF_PROMPT, NLIClusterer, mean_token_entropy,
                               parse_confidence, semantic_entropy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--model-id", default=M.DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=10, help="samples for semantic entropy")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--8bit", dest="int8", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    d = pathlib.Path(a.out) / a.dataset
    recs = [json.loads(l) for l in open(d / "generations.jsonl")]
    tok, model = M.load(a.model_id, load_in_8bit=a.int8)
    nli = NLIClusterer()

    rows = []
    for r in tqdm(recs):
        samples = M.sample_answers(model, tok, r["question"], k=a.k, think=a.think)
        se = semantic_entropy(nli.cluster(r["question"], samples))
        conf_txt = M.generate_text(model, tok, CONF_PROMPT.format(q=r["question"], a=r["answer"]), think=False)
        conf = parse_confidence(conf_txt)
        rows.append({
            "id": r["id"], "correct": r["correct"],
            "token_entropy": mean_token_entropy(r["entropies"], r["think_end_idx"]),
            "semantic_entropy": se,
            "verbalized_conf": conf if conf is not None else 0.5,
        })
    with open(d / "baselines.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    y = np.array([r["correct"] for r in rows])
    res = {
        "accuracy": float(y.mean()),
        "auroc_token_entropy": auroc_error_detection([r["token_entropy"] for r in rows], y),
        "auroc_semantic_entropy": auroc_error_detection([r["semantic_entropy"] for r in rows], y),
        "auroc_verbalized": auroc_error_detection([1 - r["verbalized_conf"] for r in rows], y),
        "ece_verbalized": ece([r["verbalized_conf"] for r in rows], y),
    }
    json.dump(res, open(d / "baseline_metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
