# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from solar.benchmark import clock_lock
from solar.benchmark.scoring import calculate_sol_score
from solar.benchmark.timing import (
    AdaptiveTimer,
    TimingPolicy,
    TorchCacheController,
    UnstableTimingError,
)
from solar.benchmark.evaluator import _ShiftingInputPool


class FakeClock:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.calls = 0

    def synchronize(self):
        pass

    def measure(self, fn):
        fn()
        self.calls += 1
        return next(self.samples)


def test_standard_policy_meets_time_and_sample_thresholds():
    clock = FakeClock([10.0] * 200)
    calls = 0

    def run():
        nonlocal calls
        calls += 1

    result = AdaptiveTimer(TimingPolicy.for_name("standard"), clock).measure(run)
    assert result.stable
    assert len(result.samples_ms) == 100
    assert clock.calls == 120  # 200 ms warmup + 1000 ms sampling
    assert calls == 121  # one initialization is not timed


def test_fast_kernel_uses_blocked_samples_to_reach_one_second():
    executions = 0

    def run():
        nonlocal executions
        executions += 1

    class ScalingClock:
        def synchronize(self):
            pass

        def measure(self, fn):
            before = executions
            fn()
            return (executions - before) * 0.05

    result = AdaptiveTimer(TimingPolicy.for_name("standard"), ScalingClock()).measure(
        run
    )
    assert result.batch_size == 20
    assert result.gpu_time_ms >= 1000
    assert len(result.samples_ms) == 1000
    assert result.p50_ms == pytest.approx(0.05)


def test_cold_cache_disables_blocking_and_clears_before_every_timed_call():
    executions = 0
    cache_clears = 0

    def run():
        nonlocal executions
        executions += 1

    def clear_cache():
        nonlocal cache_clears
        cache_clears += 1

    policy = TimingPolicy("cold", 1, 0.05, 4, 0.2, 100, 100.0, None)
    result = AdaptiveTimer(policy, FakeClock([0.05] * 20)).measure(
        run, clear_cache=clear_cache, allow_batching=False
    )

    assert result.batch_size == 1
    assert cache_clears == 5  # one warmup and four samples
    assert executions == cache_clears + 1  # plus untimed initialization


def test_official_policy_uses_five_independent_windows():
    clock = FakeClock([30.0] * 200)
    result = AdaptiveTimer(TimingPolicy.for_name("official"), clock).measure(
        lambda: None
    )
    assert len(result.window_medians_ms) == 5
    assert len(result.samples_ms) == 100
    assert result.p50_ms == 30.0


def test_quick_policy_is_not_publishable():
    policy = TimingPolicy.for_name("quick")
    assert not policy.publishable
    result = AdaptiveTimer(policy, FakeClock([5.0] * 100)).measure(lambda: None)
    assert len(result.samples_ms) == 20


def test_unstable_measurement_is_not_silently_filtered():
    policy = TimingPolicy("test", 1, 1, 4, 4, 8, 8, 0.05)
    clock = FakeClock([1.0] + [0.5, 1.5] * 4)
    with pytest.raises(UnstableTimingError) as caught:
        AdaptiveTimer(policy, clock).measure(lambda: None)
    assert len(caught.value.statistics.samples_ms) == 8


def test_timer_validates_outputs_after_timed_device_work():
    calls = 0

    def stateful_candidate():
        nonlocal calls
        calls += 1
        return 1 if calls <= 3 else 0

    policy = TimingPolicy("test", 1, 1, 3, 3, 20, 20, None)

    with pytest.raises(AssertionError, match="timed output"):
        AdaptiveTimer(policy, FakeClock([1.0] * 20)).measure(
            stateful_candidate,
            validate=lambda _args, output: (
                None
                if output == 1
                else (_ for _ in ()).throw(AssertionError("invalid timed output"))
            ),
        )


def test_shifting_pool_never_reuses_tensor_addresses():
    import torch

    seed = 0

    def factory():
        nonlocal seed
        seed += 1
        return (torch.full((8,), seed, dtype=torch.float16),)

    pool = _ShiftingInputPool(factory, max_calls=8)
    pointers = []
    for expected in range(1, 9):
        args = pool.next()
        pointers.append(args[0].data_ptr())
        pristine = pool.take_pristine(args)
        assert torch.equal(args[0], pristine[0])
        assert args[0][0].item() == expected

    assert len(pointers) == len(set(pointers))


def test_sol_score_and_audit_bounds():
    assert calculate_sol_score(3.0, 2.0, 1.0) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="baseline"):
        calculate_sol_score(1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="candidate"):
        calculate_sol_score(3.0, 0.5, 1.0)


@pytest.mark.parametrize("field", ["perf_level", "performance_level"])
def test_amd_smi_clock_level_json_variants(monkeypatch, field):
    payload = {"gpu_data": [{"gpu": 0, field: "AMDSMI_DEV_PERF_LEVEL_AUTO"}]}
    monkeypatch.setattr(clock_lock, "_amd_smi_executable", lambda: "amd-smi")
    monkeypatch.setattr(
        clock_lock.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    assert clock_lock.query_clock_level() == ("AMDSMI_DEV_PERF_LEVEL_AUTO",)


def test_cache_controller_evicts_twice_declared_last_level_cache():
    controller = TorchCacheController(32, device="cpu")
    assert controller._buffer.numel() == 64
