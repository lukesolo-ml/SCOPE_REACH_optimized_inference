"""Unit tests for scoring logic in quick_sco_re.scoring."""

import math

import numpy as np
import pytest

from quick_sco_re.scoring import _extract_token_logprobs


# ---------------------------------------------------------------------------
# _extract_token_logprobs
# ---------------------------------------------------------------------------


class TestExtractTokenLogprobs:
    """Tests for logprob extraction from SGLang output dicts."""

    TARGET_ID = 10

    def _make_output(self, position_entries, source="output"):
        key = (
            "output_token_ids_logprobs"
            if source == "output"
            else "input_token_ids_logprobs"
        )
        return {"meta_info": {key: position_entries}}

    def test_extracts_correct_logprobs(self):
        entries = [
            [(-0.5, self.TARGET_ID, "x"), (-1.0, 20, "y")],
            [(-0.3, self.TARGET_ID, "x"), (-2.0, 30, "y")],
        ]
        result = _extract_token_logprobs(
            self._make_output(entries), self.TARGET_ID
        )
        assert result == [-0.5, -0.3]

    def test_missing_target_token_skips_position(self):
        entries = [
            [(-0.5, 20, "x"), (-1.0, 30, "y")],  # no target
            [(-0.3, self.TARGET_ID, "x")],
        ]
        result = _extract_token_logprobs(
            self._make_output(entries), self.TARGET_ID
        )
        assert result == [-0.3]

    def test_none_entries_skipped(self):
        entries = [None, [(-0.5, self.TARGET_ID, "x")], None]
        result = _extract_token_logprobs(
            self._make_output(entries), self.TARGET_ID
        )
        assert result == [-0.5]

    def test_skip_parameter(self):
        entries = [
            [(-0.1, self.TARGET_ID, "x")],  # skipped
            [(-0.2, self.TARGET_ID, "x")],  # skipped
            [(-0.3, self.TARGET_ID, "x")],
        ]
        result = _extract_token_logprobs(
            self._make_output(entries), self.TARGET_ID, skip=2
        )
        assert result == [-0.3]

    def test_empty_meta_info(self):
        result = _extract_token_logprobs({}, self.TARGET_ID)
        assert result == []

    def test_empty_entries_list(self):
        result = _extract_token_logprobs(
            self._make_output([]), self.TARGET_ID
        )
        assert result == []

    def test_input_source(self):
        entries = [[(-0.7, self.TARGET_ID, "x")]]
        result = _extract_token_logprobs(
            self._make_output(entries, source="input"),
            self.TARGET_ID,
            source="input",
        )
        assert result == [-0.7]


# ---------------------------------------------------------------------------
# Score formula verification (math, no engine needed)
# ---------------------------------------------------------------------------


class TestScoreFormulas:
    """Verify M1 (SCOPE) and M2 (REACH) score math independently."""

    def test_m1_scope_is_sum_of_probs(self):
        logprobs = [-0.5, -1.0, -0.1]
        probs = np.clip(np.exp(logprobs), 0.0, 1.0)
        score = float(np.sum(probs))
        expected = math.exp(-0.5) + math.exp(-1.0) + math.exp(-0.1)
        assert abs(score - expected) < 1e-10

    def test_m2_reach_is_complement_product(self):
        logprobs = [-0.5, -1.0, -0.1]
        probs = np.clip(np.exp(logprobs), 0.0, 1.0)
        score = float(1.0 - np.prod(1.0 - probs))
        # Manual: 1 - (1-e^-0.5)(1-e^-1)(1-e^-0.1)
        manual = 1.0 - (
            (1 - math.exp(-0.5))
            * (1 - math.exp(-1.0))
            * (1 - math.exp(-0.1))
        )
        assert abs(score - manual) < 1e-10

    def test_m1_single_prob(self):
        probs = np.array([0.8])
        assert float(np.sum(probs)) == pytest.approx(0.8)

    def test_m2_single_prob(self):
        probs = np.array([0.8])
        score = float(1.0 - np.prod(1.0 - probs))
        assert score == pytest.approx(0.8)

    def test_m2_zero_probs(self):
        probs = np.array([0.0, 0.0, 0.0])
        score = float(1.0 - np.prod(1.0 - probs))
        assert score == pytest.approx(0.0)

    def test_m2_all_one_probs(self):
        probs = np.array([1.0, 1.0])
        score = float(1.0 - np.prod(1.0 - probs))
        assert score == pytest.approx(1.0)

    def test_m1_empty_probs(self):
        probs = np.array([])
        assert float(np.sum(probs)) == 0.0

    def test_m2_empty_probs(self):
        probs = np.array([])
        # prod of empty = 1.0, so 1 - 1 = 0
        assert float(1.0 - np.prod(1.0 - probs)) == pytest.approx(0.0)
