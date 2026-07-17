# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from solar.benchmark.scoring import calculate_sol_score
from solar.benchmark.timing import AdaptiveTimer, TimingPolicy, UnstableTimingError


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


def test_sol_score_and_audit_bounds():
    assert calculate_sol_score(3.0, 2.0, 1.0) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="baseline"):
        calculate_sol_score(1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="candidate"):
        calculate_sol_score(3.0, 0.5, 1.0)
