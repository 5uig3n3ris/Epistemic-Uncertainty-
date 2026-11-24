"""Generate greedy answers, label correctness, cache last-prompt-token hidden states."""
import argparse
import json
import pathlib

import numpy as np
from tqdm import tqdm

from r1epi import model as M
from r1epi.data import load_qa
from r1epi.grading import is_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa", choices=["triviaqa", "popqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model-id", default=M.DEFAULT_MODEL)
    ap.add_argument("--think", action="store_true", help="let the model produce a <think> trace")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--8bit", dest="int8", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = pathlib.Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    tok, model = M.load(a.model_id, load_in_8bit=a.int8)
    rows = load_qa(a.dataset, a.n)

    hidden = []
    with open(out / "generations.jsonl", "w") as f:
        for r in tqdm(rows):
            g = M.generate_one(model, tok, r["question"], think=a.think, max_new_tokens=a.max_new_tokens)
            hidden.append(g.pop("hidden"))
            rec = {**r, **g, "correct": is_correct(g["answer"], r["answers"])}
            f.write(json.dumps(rec) + "\n")
    np.save(out / "hidden_last_prompt_token.npy", np.stack(hidden).astype(np.float16))
    print(f"Saved {len(rows)} generations to {out}")


if __name__ == "__main__":
    main()
