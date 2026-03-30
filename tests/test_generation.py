"""Unit tests for trajectory generation helpers in quick_sco_re.generation."""

import pytest
import torch

from quick_sco_re.generation import DeferredTimeHorizonProcessor


# ---------------------------------------------------------------------------
# DeferredTimeHorizonProcessor
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for an SGLang request object."""

    def __init__(self, output_ids):
        self.output_ids = output_ids


class TestDeferredTimeHorizonProcessor:
    """Tests for the custom logit processor used for time-horizon stopping."""

    TIME_MAP = {50: 10.0, 51: 60.0, 52: 360.0}
    TRUNC_ID = 99

    def _make_param(self, output_ids, time_horizon=100.0, check_interval=5):
        return {
            "__req__": _FakeRequest(output_ids),
            "time_horizon": time_horizon,
            "trunc_id": self.TRUNC_ID,
            "time_token_map": self.TIME_MAP,
            "check_interval": check_interval,
        }

    def test_noop_before_check_interval(self):
        """Processor should not modify logits before check_interval tokens."""
        proc = DeferredTimeHorizonProcessor()
        logits = torch.zeros(1, 200)

        # 3 tokens generated, check_interval=5 -> no-op
        param = self._make_param([52, 52, 52], check_interval=5)
        result = proc(logits, [param])

        assert torch.all(result == 0.0)

    def test_fires_when_horizon_exceeded(self):
        """Processor forces trunc_id when elapsed time >= horizon."""
        proc = DeferredTimeHorizonProcessor()
        vocab_size = 200
        logits = torch.zeros(1, vocab_size)

        # 5 tokens, check_interval=5 -> checks on step 5
        # token 52 = 360 min, way over 100 min horizon
        output_ids = [52, 1, 1, 1, 1]
        param = self._make_param(output_ids, time_horizon=100.0, check_interval=5)
        result = proc(logits, [param])

        assert result[0, self.TRUNC_ID] == 0.0
        # All other logits should be -inf
        mask = torch.ones(vocab_size, dtype=torch.bool)
        mask[self.TRUNC_ID] = False
        assert torch.all(result[0, mask] == float("-inf"))

    def test_does_not_fire_under_horizon(self):
        """Processor leaves logits unchanged when under time horizon."""
        proc = DeferredTimeHorizonProcessor()
        logits = torch.zeros(1, 200)

        # 5 tokens, only token 50 = 10 min, well under 100 min
        output_ids = [50, 1, 1, 1, 1]
        param = self._make_param(output_ids, time_horizon=100.0, check_interval=5)
        result = proc(logits, [param])

        assert torch.all(result == 0.0)

    def test_incremental_cursor(self):
        """Processor maintains cursor state across calls."""
        proc = DeferredTimeHorizonProcessor()
        logits = torch.zeros(1, 200)

        # First call: 5 tokens, 10 min elapsed
        param = self._make_param([50, 1, 1, 1, 1], time_horizon=100.0, check_interval=5)
        proc(logits, [param])

        assert param["_cursor"] == 5
        assert param["_elapsed"] == 10.0

        # Second call: 5 more tokens (10 total), add another 60 min -> 70 total
        param["__req__"] = _FakeRequest([50, 1, 1, 1, 1, 51, 1, 1, 1, 1])
        logits = torch.zeros(1, 200)
        proc(logits, [param])

        assert param["_cursor"] == 10
        assert param["_elapsed"] == 70.0
        # Still under 100 -> logits untouched
        assert torch.all(logits == 0.0)

    def test_skips_entry_without_req(self):
        """Processor gracefully skips param dicts without __req__."""
        proc = DeferredTimeHorizonProcessor()
        logits = torch.zeros(1, 200)
        result = proc(logits, [{"check_interval": 1}])
        assert torch.all(result == 0.0)

    def test_non_check_step_is_noop(self):
        """On steps between check intervals, processor is a no-op."""
        proc = DeferredTimeHorizonProcessor()
        logits = torch.zeros(1, 200)

        # 7 tokens, check_interval=5: checks at 5, skips 6, skips 7
        output_ids = [52] * 7  # way over any horizon
        param = self._make_param(output_ids, time_horizon=1.0, check_interval=5)

        # First call at n_generated=7: 7 % 5 != 0 -> no-op
        result = proc(logits, [param])
        # But wait, len(output_ids)=7, check_interval=5: 7 >= 5 is True, 7 % 5 = 2 != 0 -> skip
        assert torch.all(result == 0.0)

    def test_batch_processing(self):
        """Processor handles multiple requests in a batch."""
        proc = DeferredTimeHorizonProcessor()
        logits = torch.zeros(2, 200)

        # Request 0: over horizon, Request 1: under horizon
        param0 = self._make_param([52, 1, 1, 1, 1], time_horizon=100.0, check_interval=5)
        param1 = self._make_param([50, 1, 1, 1, 1], time_horizon=100.0, check_interval=5)

        result = proc(logits, [param0, param1])

        # Request 0 should be forced to trunc_id
        assert result[0, self.TRUNC_ID] == 0.0
        # Request 1 should be untouched
        assert torch.all(result[1] == 0.0)
