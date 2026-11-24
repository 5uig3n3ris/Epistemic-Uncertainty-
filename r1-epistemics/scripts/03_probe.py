"""Layer-wise linear probes vs. black-box baselines."""
import argparse
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from r1epi.probing import layerwise_auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    d = pathlib.Path(a.out) / a.dataset
    X = np.load(d / "hidden_last_prompt_token.npy")
    y = np.array([json.loads(l)["correct"] for l in open(d / "generations.jsonl")]).astype(int)
    print(f"N={len(y)}, accuracy={y.mean():.3f}, hidden={X.shape}")

    au = layerwise_auroc(X, y)
    best = int(au.argmax())
    print(f"Best layer {best}: AUROC {au[best]:.3f}")

    plt.figure(figsize=(6, 3.5))
    plt.plot(au, label="linear probe (CV)")
    bm = d / "baseline_metrics.json"
    if bm.exists():
        m = json.load(open(bm))
        plt.axhline(m["auroc_semantic_entropy"], ls="--", c="C1", label="semantic entropy")
        plt.axhline(m["auroc_token_entropy"], ls=":", c="C2", label="token entropy")
    plt.xlabel("layer"); plt.ylabel("AUROC (error detection)"); plt.legend(); plt.tight_layout()
    plt.savefig(d / "probe_vs_baselines.png", dpi=150)
    json.dump({"layer_auroc": au.tolist(), "best_layer": best}, open(d / "probe_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
