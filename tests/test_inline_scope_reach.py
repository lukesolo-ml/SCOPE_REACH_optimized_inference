"""Unit tests for inline SCOPE/REACH estimation.

Covers:
  1. Pure-math REACH recursion
  2. Pure-math SCOPE truncated sum
  3. Processor unit tests with fake requests
  4. Multi-request batch isolation
  5. Early-exit when all tracked tokens have occurred
  6. Round-trip persistence with and without inline fields
"""

import numpy as np
import pytest
import torch

from quick_sco_re.generation import (
    InlineScopeReachProcessor,
    ChainedProcessor,
    DeferredTimeHorizonProcessor,
    _RESULTS,
    _store_result,
    pop_inline_result,
)
from quick_sco_re.io import save_trajectories, load_trajectories
from quick_sco_re.structures import GeneratedTrajectory, TrajectoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for an SGLang request object."""

    def __init__(self, output_ids):
        self.output_ids = output_ids


def _make_sr_param(output_ids, tracked_ids, request_id="test-req"):
    return {
        "__req__": _FakeRequest(output_ids),
        "tracked_ids": tracked_ids,
        "request_id": request_id,
    }


def _logits_from_probs(probs_list, vocab_size=100):
    """Create logits tensor such that softmax(logits)[tracked_ids] == probs.

    probs_list is a full-vocab probability vector (length vocab_size).
    """
    probs = torch.tensor(probs_list, dtype=torch.float64)
    # Clamp to avoid log(0)
    probs = probs.clamp(min=1e-30)
    logits = probs.log().float().unsqueeze(0)  # shape (1, V)
    return logits


# ---------------------------------------------------------------------------
# 1. Pure-math REACH recursion
# ---------------------------------------------------------------------------


class TestReachMath:
    """Verify REACH = 1 - prod(1 - p_i) matches the incremental update."""

    def test_known_sequence(self):
        probs = [0.1, 0.2, 0.3, 0.05, 0.15]
        # Brute force
        expected = 1.0 - np.prod([1.0 - p for p in probs])
        # Incremental
        reach = 0.0
        for p in probs:
            reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach - expected) < 1e-12

    def test_single_prob(self):
        reach = 0.0
        p = 0.7
        reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach - 0.7) < 1e-12

    def test_zero_probs(self):
        reach = 0.0
        for p in [0.0, 0.0, 0.0]:
            reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach) < 1e-12

    def test_all_one_probs(self):
        reach = 0.0
        for p in [1.0, 1.0]:
            reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 2. Pure-math SCOPE truncated sum
# ---------------------------------------------------------------------------


class TestScopeMath:
    """Verify SCOPE = sum of p_k at pre-occurrence steps."""

    def test_no_occurrence(self):
        """All steps contribute when the token never occurs."""
        probs = [0.1, 0.2, 0.3]
        expected = sum(probs)
        scope = sum(probs)  # No truncation
        assert abs(scope - expected) < 1e-12

    def test_with_occurrence_at_step_2(self):
        """Occurrence at step 2 means steps 0, 1, 2 contribute (inclusive)."""
        # Steps: 0 (p=0.1), 1 (p=0.2), 2 (p=0.3 - occurrence step), 3 (p=0.4 - skipped)
        probs = [0.1, 0.2, 0.3, 0.4]
        occurrence_step = 2
        expected = sum(probs[: occurrence_step + 1])  # 0.1 + 0.2 + 0.3
        assert abs(expected - 0.6) < 1e-12

    def test_occurrence_at_first_step(self):
        """Token occurs immediately — only step 0's prob contributes."""
        probs = [0.5, 0.3, 0.2]
        expected = 0.5
        scope = probs[0]
        assert abs(scope - expected) < 1e-12


# ---------------------------------------------------------------------------
# 3. Processor unit test with fake requests
# ---------------------------------------------------------------------------


class TestInlineScopeReachProcessor:
    """Test processor mechanics with synthetic logits and fake requests."""

    def test_lazy_init(self):
        proc = InlineScopeReachProcessor()
        tracked_ids = [5, 10, 15]
        param = _make_sr_param([], tracked_ids)

        # Build logits where tracked tokens have specific probabilities
        vocab_size = 20
        probs = np.full(vocab_size, 1e-6)
        probs[5] = 0.1
        probs[10] = 0.2
        probs[15] = 0.05
        probs = probs / probs.sum()
        logits = _logits_from_probs(probs.tolist(), vocab_size)

        proc(logits, [param])

        assert "_sr_cursor" in param
        assert "_sr_scope" in param
        assert "_sr_reach" in param
        assert param["_sr_cursor"] == 0
        assert len(param["_sr_scope"]) == 3
        assert len(param["_sr_reach"]) == 3

    def test_scope_reach_update_single_step(self):
        proc = InlineScopeReachProcessor()
        tracked_ids = [5, 10]
        param = _make_sr_param([], tracked_ids, request_id="req1")

        vocab_size = 20
        probs = np.full(vocab_size, 0.01)
        probs[5] = 0.3
        probs[10] = 0.2
        probs = probs / probs.sum()
        logits = _logits_from_probs(probs.tolist(), vocab_size)

        proc(logits, [param])

        result = pop_inline_result("req1")
        assert result is not None
        # After one step, scope ≈ prob, reach ≈ prob
        np.testing.assert_allclose(result["scope"][0], probs[5], rtol=1e-4)
        np.testing.assert_allclose(result["scope"][1], probs[10], rtol=1e-4)
        np.testing.assert_allclose(result["reach"][0], probs[5], rtol=1e-4)
        np.testing.assert_allclose(result["reach"][1], probs[10], rtol=1e-4)

    def test_occurred_flag_detection(self):
        """After a tracked token appears in output_ids, its flag flips."""
        proc = InlineScopeReachProcessor()
        tracked_ids = [5, 10]

        # Step 0: no output yet
        param = _make_sr_param([], tracked_ids, request_id="req2")
        vocab_size = 20
        probs = np.full(vocab_size, 0.01)
        probs[5] = 0.3
        probs[10] = 0.2
        probs = probs / probs.sum()
        logits = _logits_from_probs(probs.tolist(), vocab_size)
        proc(logits, [param])

        assert not param["_sr_occurred_flag"][0]
        assert not param["_sr_occurred_flag"][1]

        # Step 1: token 5 was sampled at step 0
        param["__req__"] = _FakeRequest([5])
        logits = _logits_from_probs(probs.tolist(), vocab_size)
        proc(logits, [param])

        assert param["_sr_occurred_flag"][0]  # token 5 occurred
        assert not param["_sr_occurred_flag"][1]  # token 10 not yet
        assert param["_sr_occurred_index"][0] == 0

    def test_no_update_after_occurrence(self):
        """Once a token has occurred, its scope/reach should not change."""
        proc = InlineScopeReachProcessor()
        tracked_ids = [5]

        param = _make_sr_param([], tracked_ids, request_id="req3")
        vocab_size = 20
        probs = np.full(vocab_size, 0.01)
        probs[5] = 0.3
        probs = probs / probs.sum()
        logits = _logits_from_probs(probs.tolist(), vocab_size)

        # Step 0
        proc(logits, [param])
        scope_after_step0 = param["_sr_scope"][0]
        reach_after_step0 = param["_sr_reach"][0]

        # Step 1: token 5 sampled at step 0
        param["__req__"] = _FakeRequest([5])
        proc(logits, [param])
        scope_after_step1 = param["_sr_scope"][0]
        reach_after_step1 = param["_sr_reach"][0]

        # scope should have been updated at step 1 (before flag flip on step 2)
        # but reach should not change at the occurrence step per the plan

        # Step 2: another non-tracked token
        param["__req__"] = _FakeRequest([5, 7])
        proc(logits, [param])
        scope_after_step2 = param["_sr_scope"][0]
        reach_after_step2 = param["_sr_reach"][0]

        # After occurrence, no further updates
        assert scope_after_step2 == scope_after_step1
        assert reach_after_step2 == reach_after_step1

    def test_returns_logits_unmodified(self):
        """Processor must not modify logits."""
        proc = InlineScopeReachProcessor()
        tracked_ids = [5]
        param = _make_sr_param([], tracked_ids)

        logits = torch.randn(1, 50)
        logits_copy = logits.clone()
        proc(logits, [param])

        torch.testing.assert_close(logits, logits_copy)

    def test_skips_without_req(self):
        proc = InlineScopeReachProcessor()
        logits = torch.zeros(1, 20)
        result = proc(logits, [{"tracked_ids": [5]}])
        assert torch.all(result == 0.0)

    def test_skips_without_tracked_ids(self):
        proc = InlineScopeReachProcessor()
        logits = torch.zeros(1, 20)
        result = proc(logits, [{"__req__": _FakeRequest([])}])
        assert torch.all(result == 0.0)


# ---------------------------------------------------------------------------
# 4. Multi-request batch isolation
# ---------------------------------------------------------------------------


class TestBatchIsolation:
    """Two concurrent requests with different tracked_ids must not cross-contaminate."""

    def test_no_cross_contamination(self):
        proc = InlineScopeReachProcessor()
        # Clear any leftover results
        _RESULTS.clear()

        vocab_size = 20
        probs = np.full(vocab_size, 0.01)
        probs[5] = 0.3
        probs[10] = 0.2
        probs = probs / probs.sum()
        logits = _logits_from_probs(probs.tolist(), vocab_size)
        # Expand to batch of 2
        logits = logits.repeat(2, 1)

        param0 = _make_sr_param([], [5], request_id="batch-0")
        param1 = _make_sr_param([], [10], request_id="batch-1")

        proc(logits, [param0, param1])

        r0 = pop_inline_result("batch-0")
        r1 = pop_inline_result("batch-1")

        assert r0 is not None and r1 is not None
        # Request 0 tracks token 5, request 1 tracks token 10
        assert len(r0["scope"]) == 1
        assert len(r1["scope"]) == 1
        np.testing.assert_allclose(r0["scope"][0], probs[5], rtol=1e-4)
        np.testing.assert_allclose(r1["scope"][0], probs[10], rtol=1e-4)


# ---------------------------------------------------------------------------
# 5. Early-exit test
# ---------------------------------------------------------------------------


class TestEarlyExit:
    """When all tracked tokens have occurred, softmax should be skipped."""

    def test_no_further_updates_after_all_occurred(self):
        proc = InlineScopeReachProcessor()
        tracked_ids = [5]
        vocab_size = 20
        probs = np.full(vocab_size, 0.01)
        probs[5] = 0.3
        probs = probs / probs.sum()

        # Step 0: no output yet
        param = _make_sr_param([], tracked_ids, request_id="early-exit")
        logits = _logits_from_probs(probs.tolist(), vocab_size)
        proc(logits, [param])
        scope_step0 = param["_sr_scope"][0]

        # Step 1: token 5 was sampled, flag flips
        param["__req__"] = _FakeRequest([5])
        logits = _logits_from_probs(probs.tolist(), vocab_size)
        proc(logits, [param])
        scope_step1 = param["_sr_scope"][0]

        # Step 2: all tokens occurred, early exit — scope should not change
        param["__req__"] = _FakeRequest([5, 7])
        logits = _logits_from_probs(probs.tolist(), vocab_size)
        proc(logits, [param])
        scope_step2 = param["_sr_scope"][0]

        assert scope_step2 == scope_step1


# ---------------------------------------------------------------------------
# 6. Chained processor
# ---------------------------------------------------------------------------


class TestChainedProcessor:
    """ChainedProcessor dispatches to sub-processors correctly."""

    def test_scope_reach_only(self):
        proc = ChainedProcessor()
        tracked_ids = [5]
        param = _make_sr_param([], tracked_ids, request_id="chain-sr")
        vocab_size = 20
        probs = np.full(vocab_size, 0.01)
        probs[5] = 0.3
        probs = probs / probs.sum()
        logits = _logits_from_probs(probs.tolist(), vocab_size)

        proc(logits, [param])
        result = pop_inline_result("chain-sr")
        assert result is not None
        assert len(result["scope"]) == 1

    def test_time_horizon_only(self):
        """Time-horizon params without tracked_ids should still work."""
        proc = ChainedProcessor()
        param = {
            "__req__": _FakeRequest([50, 1, 1, 1, 1]),
            "time_horizon": 100.0,
            "trunc_id": 99,
            "time_token_map": {50: 360.0},
            "check_interval": 5,
        }
        logits = torch.zeros(1, 200)
        result = proc(logits, [param])
        # Should have forced trunc_id
        assert result[0, 99] == 0.0

    def test_both_processors(self):
        """Both time-horizon and scope/reach should fire."""
        proc = ChainedProcessor()
        param = {
            "__req__": _FakeRequest([50, 1, 1, 1, 1]),
            "time_horizon": 100.0,
            "trunc_id": 99,
            "time_token_map": {50: 360.0},
            "check_interval": 5,
            "tracked_ids": [5],
            "request_id": "chain-both",
        }
        vocab_size = 200
        logits = torch.zeros(1, vocab_size)
        result = proc(logits, [param])

        # Time horizon should have fired
        assert result[0, 99] == 0.0
        # SCOPE/REACH should have a result (but values are meaningless since
        # logits were zeroed by time-horizon first — just verify it ran)
        sr = pop_inline_result("chain-both")
        assert sr is not None


# ---------------------------------------------------------------------------
# 7. Round-trip persistence
# ---------------------------------------------------------------------------


class TestPersistenceRoundtrip:
    """Save and load trajectories with inline SCOPE/REACH fields."""

    def test_roundtrip_with_inline_sr(self, tmp_path):
        K = 3
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2, 3],
                timeline_terminating_id=3,
                scope_estimates=np.array([0.5, 0.3, 0.1]),
                reach_estimates=np.array([0.4, 0.25, 0.09]),
                occurred_flag=np.array([True, False, False]),
                occurred_index=np.array([2, -1, -1], dtype=np.int64),
                inline_tracked_ids=[10, 20, 30],
                inline_tracked_name="test_set",
            ),
            GeneratedTrajectory(
                patient_idx=1, sample_idx=0, traj_type=TrajectoryType.M2,
                prompt_len=15, output_ids=[4, 5],
                timeline_terminating_id=None,
                scope_estimates=np.array([0.7, 0.2, 0.0]),
                reach_estimates=np.array([0.6, 0.18, 0.0]),
                occurred_flag=np.array([True, True, False]),
                occurred_index=np.array([0, 1, -1], dtype=np.int64),
                inline_tracked_ids=[10, 20, 30],
                inline_tracked_name="test_set",
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)

        assert len(loaded) == 2
        for orig, ld in zip(trajs, loaded):
            assert ld.patient_idx == orig.patient_idx
            assert ld.output_ids == orig.output_ids
            np.testing.assert_array_almost_equal(ld.scope_estimates, orig.scope_estimates)
            np.testing.assert_array_almost_equal(ld.reach_estimates, orig.reach_estimates)
            np.testing.assert_array_equal(ld.occurred_flag, orig.occurred_flag)
            np.testing.assert_array_equal(ld.occurred_index, orig.occurred_index)
            assert ld.inline_tracked_ids == orig.inline_tracked_ids
            assert ld.inline_tracked_name == orig.inline_tracked_name

    def test_roundtrip_without_inline_sr(self, tmp_path):
        """Trajectories without inline SR should load with None fields."""
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2, 3],
                timeline_terminating_id=3,
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)

        assert loaded[0].scope_estimates is None
        assert loaded[0].reach_estimates is None
        assert loaded[0].occurred_flag is None
        assert loaded[0].occurred_index is None
        assert loaded[0].inline_tracked_ids is None
        assert loaded[0].inline_tracked_name is None

    def test_mixed_trajectories(self, tmp_path):
        """Mix of trajectories with and without inline SR."""
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2],
                timeline_terminating_id=2,
                scope_estimates=np.array([0.5]),
                reach_estimates=np.array([0.4]),
                occurred_flag=np.array([True]),
                occurred_index=np.array([1], dtype=np.int64),
                inline_tracked_ids=[10],
                inline_tracked_name="mixed",
            ),
            GeneratedTrajectory(
                patient_idx=1, sample_idx=0, traj_type=TrajectoryType.M2,
                prompt_len=10, output_ids=[3, 4],
                timeline_terminating_id=None,
                # No inline SR for this one
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)

        assert loaded[0].scope_estimates is not None
        np.testing.assert_array_almost_equal(loaded[0].scope_estimates, [0.5])
        # Second trajectory has no inline SR data — offsets are equal, so None
        assert loaded[1].scope_estimates is None


# ---------------------------------------------------------------------------
# 8. Result store
# ---------------------------------------------------------------------------


class TestResultStore:
    """Test module-level _RESULTS dict behavior."""

    def test_store_and_pop(self):
        _RESULTS.clear()
        _store_result("r1", {"scope": np.array([1.0])})
        result = pop_inline_result("r1")
        assert result is not None
        assert pop_inline_result("r1") is None  # Already popped

    def test_pop_missing_returns_none(self):
        _RESULTS.clear()
        assert pop_inline_result("nonexistent") is None

    def test_lru_eviction(self):
        _RESULTS.clear()
        from quick_sco_re.generation import _RESULTS_MAX_SIZE
        # Store more than max
        for i in range(_RESULTS_MAX_SIZE + 10):
            _store_result(f"r{i}", {"val": i})
        assert len(_RESULTS) == _RESULTS_MAX_SIZE
        # Oldest should have been evicted
        assert pop_inline_result("r0") is None
        assert pop_inline_result("r9") is None
        # Newest should still be there
        assert pop_inline_result(f"r{_RESULTS_MAX_SIZE + 9}") is not None
        _RESULTS.clear()
