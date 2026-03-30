"""Unit tests for time-based truncation logic in quick_sco_re.generation."""

import pytest

from quick_sco_re.generation import apply_time_truncation


# ---------------------------------------------------------------------------
# apply_time_truncation
# ---------------------------------------------------------------------------


class TestApplyTimeTruncation:
    """Tests for the post-hoc exact time truncation walk."""

    TOKEN_MAP = {50: 10.0, 51: 60.0, 52: 360.0}

    def test_truncates_at_exact_boundary(self):
        # 10 + 60 = 70 < 100, then 360 pushes to 430 >= 100
        output_ids = [1, 50, 2, 51, 3, 52, 4]
        trimmed, was_truncated, idx = apply_time_truncation(
            output_ids, self.TOKEN_MAP, max_time=100.0
        )
        assert was_truncated is True
        assert idx == 5  # token 52 is at index 5
        assert trimmed == [1, 50, 2, 51, 3]

    def test_no_truncation_under_horizon(self):
        output_ids = [1, 50, 2, 50, 3]  # 10 + 10 = 20 < 100
        trimmed, was_truncated, idx = apply_time_truncation(
            output_ids, self.TOKEN_MAP, max_time=100.0
        )
        assert was_truncated is False
        assert idx is None
        assert trimmed == output_ids

    def test_empty_input(self):
        trimmed, was_truncated, idx = apply_time_truncation(
            [], self.TOKEN_MAP, max_time=100.0
        )
        assert was_truncated is False
        assert idx is None
        assert trimmed == []

    def test_first_token_exceeds_horizon(self):
        output_ids = [52, 1, 2]  # 360 >= 100 immediately
        trimmed, was_truncated, idx = apply_time_truncation(
            output_ids, self.TOKEN_MAP, max_time=100.0
        )
        assert was_truncated is True
        assert idx == 0
        assert trimmed == []

    def test_all_non_time_tokens(self):
        output_ids = [1, 2, 3, 4, 5]
        trimmed, was_truncated, idx = apply_time_truncation(
            output_ids, self.TOKEN_MAP, max_time=100.0
        )
        assert was_truncated is False
        assert trimmed == output_ids

    def test_exact_boundary_triggers(self):
        # 60 + 60 = 120, and max_time=120 -> should truncate (>=)
        output_ids = [51, 51]
        trimmed, was_truncated, idx = apply_time_truncation(
            output_ids, self.TOKEN_MAP, max_time=120.0
        )
        assert was_truncated is True
        assert idx == 1
        assert trimmed == [51]

    def test_single_time_token_under_horizon(self):
        output_ids = [50]  # 10 < 100
        trimmed, was_truncated, idx = apply_time_truncation(
            output_ids, self.TOKEN_MAP, max_time=100.0
        )
        assert was_truncated is False
        assert trimmed == [50]


