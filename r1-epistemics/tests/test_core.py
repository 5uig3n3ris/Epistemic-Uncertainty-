import numpy as np

from r1epi.calibration import auroc_error_detection, ece
from r1epi.grading import extract_answer, is_correct
from r1epi.probing import layerwise_auroc, probe_direction
from r1epi.uncertainty import mean_token_entropy, parse_confidence, semantic_entropy


def test_grading():
    assert is_correct("The Eiffel Tower.", ["Eiffel Tower"])
    assert not is_correct("Big Ben", ["Eiffel Tower"])
    assert not is_correct("", ["x"])
    assert extract_answer("<think>hmm</think>\n\nParis") == "Paris"


def test_ece_and_auroc():
    assert ece([1.0, 0.0], [1, 0]) == 0.0
    assert ece([1.0, 1.0], [0, 0]) == 1.0
    assert auroc_error_detection([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) == 1.0


def test_semantic_entropy():
    assert semantic_entropy([0, 0, 0, 0]) == 0.0
    assert abs(semantic_entropy([0, 1]) - np.log(2)) < 1e-9


def test_misc():
    assert parse_confidence("85") == 0.85
    assert parse_confidence("no idea") is None
    assert mean_token_entropy([5, 5, 1, 1], think_end_idx=1) == 1.0


def test_probe_recovers_planted_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    X = rng.normal(size=(300, 3, 16))
    X[:, 1, 0] += 3 * y  # only layer 1 carries signal
    au = layerwise_auroc(X, y)
    assert au.argmax() == 1 and au[1] > 0.9
    w = probe_direction(X[:, 1], y)
    assert abs(np.linalg.norm(w) - 1) < 1e-6 and np.argmax(np.abs(w)) == 0
