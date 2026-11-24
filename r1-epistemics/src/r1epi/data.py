"""Load QA datasets into a common format: {id, question, answers: [aliases]}."""
import json
import random


def load_qa(name: str, n: int, seed: int = 0):
    from datasets import load_dataset

    if name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        rows = [
            {"id": str(r["question_id"]), "question": r["question"], "answers": r["answer"]["aliases"]}
            for r in ds
        ]
    elif name == "popqa":
        ds = load_dataset("akariasai/PopQA", split="test")
        rows = [
            {"id": str(r["id"]), "question": r["question"], "answers": json.loads(r["possible_answers"])}
            for r in ds
        ]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    random.Random(seed).shuffle(rows)
    return rows[:n]
