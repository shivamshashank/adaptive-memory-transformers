"""Checks for the offline descriptive analysis, with no model download."""

import runpy
from pathlib import Path

analysis = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/analyze_pilot_failures.py")
)


def test_answer_validity_is_a_complete_letter_not_substring():
    valid = analysis["classify_answer"]
    assert valid(" A") and valid("D\n")
    assert not any(valid(text) for text in ("", " ", "AB", "The", "<|im_end|>", "A."))


def test_span_overlap_handles_boundaries():
    assert analysis["span_indices"]([(0, 0), (0, 3), (3, 5), (5, 8)], 3, 5) == [2]
    assert analysis["span_indices"]([(0, 3), (3, 5)], 2, 4) == [0, 1]


def test_analysis_keeps_full_failures_and_separates_validity_from_accuracy():
    data = [
        {
            "id": "x",
            "gold": "A",
            "target_fact_index": 0,
            "target_color": "red",
            "context": "Records\nThe Aster locker is red.\nThe Birch locker is blue.\n",
            "query": "What color is the Aster locker?",
            "options": ["red", "blue", "green", "yellow"],
        }
    ]
    row = {
        "example_id": "x",
        "policy": "full",
        "ratio": None,
        "correct": False,
        "prediction": "B",
        "gold": "A",
        "choice_logits": {"A": 1, "B": 3, "C": 0, "D": 0},
        "raw_greedy_text": "B",
        "retained_positions": [0, 1, 2, 3],
    }
    rows = [
        row,
        row
        | {
            "policy": "adaptive",
            "ratio": 0.5,
            "raw_greedy_text": "The",
            "retained_positions": [2, 3],
        },
        row | {"policy": "uniform", "ratio": 0.5, "retained_positions": [0, 2]},
    ]
    result = analysis["analyze"](data, rows, {"x": {"header": [0, 1], "target_fact": [2, 3]}})
    assert result["full_failures"][0]["prediction_minus_gold_logit"] == 2
    assert result["full_failures"][0]["other_positions_with_predicted_color"] == [1]
    assert result["compressed_position_zero_association"] == [
        {"position_zero_retained": False, "n": 1, "valid_next_tokens": 0},
        {"position_zero_retained": True, "n": 1, "valid_next_tokens": 1},
    ]
    assert result["groups"][2]["valid_next_tokens"] == 1
    assert result["groups"][2]["correct"] == 0
