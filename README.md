# r1epi: Does a reasoning LLM know when it's about to hallucinate?

**Model:** DeepSeek-R1-Distill-Qwen-7B (runs locally on a 16 GB GPU, fp16 or 8-bit)

## Research question

Does the model's internal state encode whether its answer is correct, *before* it answers?
When a `<think>` trace self-corrects ("wait, let me recheck"), does the internal
correctness signal change **before** the visible correction, or does the self-correction
merely look like it?

## Hypotheses (written before running experiments)

- **H1.** A linear probe on mid-layer hidden states predicts answer correctness better than
  token entropy, and is competitive with semantic entropy at a fraction of the cost.
- **H2.** Probe signal is strongest at intermediate layers and degrades in the final layers.
- **H3.** In traces with a self-correction, the probe signal drops before the correction
  tokens, i.e. the model "knows" before it says so. (Causal check: steer along the probe direction.)

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Generate, label, cache hidden states | `scripts/01_generate.py` | `results/<ds>/generations.jsonl`, `hidden_*.npy` |
| 2. Black-box baselines | `scripts/02_baselines.py` | token entropy, semantic entropy, verbalized confidence: AUROC + ECE |
| 3. Layer-wise probes | `scripts/03_probe.py` | AUROC per layer vs. baselines (plot) |
| 4. Causal tests (TODO) | | activation patching and steering |
| 5. Self-correction traces (TODO) | | per-token probe trajectory across `<think>` |

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                   # CPU-only sanity tests

# Prototype: no reasoning trace, 200 questions
python scripts/01_generate.py --dataset triviaqa --n 200
python scripts/02_baselines.py --dataset triviaqa
python scripts/03_probe.py --dataset triviaqa

# Full: with reasoning traces (slower, longer generations)
python scripts/01_generate.py --dataset triviaqa --n 1000 --think --max-new-tokens 1024
```

Add `--8bit` on tight VRAM. If memory runs out in step 2, run the NLI model on CPU (`NLIClusterer(device="cpu")`).

## Design notes

- **Probe input:** hidden state of the last prompt token at every layer, so the probe only sees what the
  model knows *before* generating. Labels come from the generated answer.
- **Correctness:** normalized alias match (`grading.py`). Noisy for free-form answers; a later step should
  add an LLM judge and report agreement.
- **Semantic entropy:** discrete version, greedy bidirectional-entailment clustering with DeBERTa-MNLI.
- **Probe eval:** 5-fold cross-validated AUROC with a regularized logistic regression. Use train/test
  splits by *question* (and across datasets) before claiming generalization.

## Results

_To be filled in._

| Method | TriviaQA AUROC | PopQA AUROC |
|---|---|---|
| Token entropy | | |
| Verbalized confidence | | |
| Semantic entropy | | |
| Linear probe (best layer) | | |

## Reading list

- Kadavath et al., *Language Models (Mostly) Know What They Know*
- Farquhar et al., *Detecting hallucinations in LLMs using semantic entropy* (Nature, 2024)
- Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying*
- Marks & Tegmark, *The Geometry of Truth*

## License

MIT
