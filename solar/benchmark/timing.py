# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adaptive ROCm device-event timing policies."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class TimingPolicy:
    name: str
    warmup_min_calls: int
    warmup_min_ms: float
    sample_min_calls: int
    sample_min_ms: float
    max_calls: int
    max_ms: float
    stability_ratio: float | None
    windows: int = 1
    publishable: bool = True

    @classmethod
    def for_name(cls, name: str) -> "TimingPolicy":
        profiles = {
            "standard": cls("standard", 10, 200.0, 30, 1000.0, 100_000, 10_000.0, 0.05),
            "official": cls(
                "official", 10, 200.0, 20, 600.0, 100_000, 10_000.0, 0.05, windows=5
            ),
            "quick": cls(
                "quick", 0, 25.0, 1, 100.0, 100_000, 100.0, None, publishable=False
            ),
        }
        try:
            return profiles[name]
        except KeyError as exc:
            raise ValueError(f"unknown timing profile: {name}") from exc


@dataclass(frozen=True)
class TimingStatistics:
    samples_ms: tuple[float, ...]
    p20_ms: float
    p50_ms: float
    p80_ms: float
    p95_ms: float
    iqr_ms: float
    mean_ms: float
    std_ms: float
    stable: bool
    batch_size: int = 1
    gpu_time_ms: float = 0.0
    window_medians_ms: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnstableTimingError(RuntimeError):
    """Raised when the maximum timing budget is exhausted without stability."""

    def __init__(self, statistics: TimingStatistics):
        super().__init__("IQR/median did not reach the required stability threshold")
        self.statistics = statistics


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    samples_ms: list[float], stability_ratio: float | None
) -> TimingStatistics:
    if not samples_ms or any(not math.isfinite(x) or x < 0 for x in samples_ms):
        raise ValueError("timing samples must be finite, non-negative, and non-empty")
    p25 = _percentile(samples_ms, 0.25)
    p50 = _percentile(samples_ms, 0.50)
    p75 = _percentile(samples_ms, 0.75)
    iqr = p75 - p25
    stable = stability_ratio is None or (p50 > 0 and iqr / p50 <= stability_ratio)
    return TimingStatistics(
        samples_ms=tuple(samples_ms),
        p20_ms=_percentile(samples_ms, 0.20),
        p50_ms=p50,
        p80_ms=_percentile(samples_ms, 0.80),
        p95_ms=_percentile(samples_ms, 0.95),
        iqr_ms=iqr,
        mean_ms=statistics.fmean(samples_ms),
        std_ms=statistics.pstdev(samples_ms),
        stable=stable,
    )


class TorchEventClock:
    """HIP device-event clock exposed through PyTorch's compatibility API."""

    def measure(self, fn: Callable[[], Any]) -> float:
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()


class AdaptiveTimer:
    """Execute one initialization, adaptive warmup, and stable sample windows."""

    def __init__(self, policy: TimingPolicy, clock: Any | None = None):
        self.policy = policy
        self.clock = clock or TorchEventClock()

    def measure(
        self,
        fn: Callable[..., Any],
        *,
        setup: Callable[[], tuple[Any, ...]] | None = None,
        clear_cache: Callable[[], None] | None = None,
        allow_batching: bool = True,
        validate: Callable[[tuple[Any, ...], Any], None] | None = None,
    ) -> TimingStatistics:
        setup = setup or (lambda: ())
        clear_cache = clear_cache or (lambda: None)

        # Initialization/JIT is intentionally outside all timing budgets.
        initial_args = setup()
        initial_output = fn(*initial_args)
        self.clock.synchronize()
        if validate is not None:
            validate(initial_args, initial_output)
        warmup_calls = 0
        warmup_ms = 0.0
        last_elapsed_ms = 0.0
        while (
            warmup_calls < self.policy.warmup_min_calls
            or warmup_ms < self.policy.warmup_min_ms
        ):
            clear_cache()
            args = setup()
            output = None

            def run_warmup() -> None:
                nonlocal output
                output = fn(*args)

            elapsed = self.clock.measure(run_warmup)
            if validate is not None:
                validate(args, output)
            last_elapsed_ms = elapsed
            warmup_calls += 1
            warmup_ms += elapsed
            if warmup_calls >= self.policy.max_calls or warmup_ms >= self.policy.max_ms:
                raise RuntimeError("warmup exceeded the timing safety budget")

        # Block very fast kernels in application-cache mode so the time budget
        # can be met without exhausting the statistical-sample cap. Each stored
        # sample is normalized back to per-call latency. Cold-cache mode keeps
        # batch_size at one so every call receives a fresh cache clear.
        batch_size = (
            max(1, math.ceil(1.0 / max(last_elapsed_ms, 1e-9)))
            if allow_batching and validate is None
            else 1
        )
        windows: list[TimingStatistics] = []
        for _ in range(self.policy.windows):
            windows.append(
                self._measure_window(
                    fn, setup, clear_cache, batch_size, validate=validate
                )
            )
        samples = [sample for window in windows for sample in window.samples_ms]
        result = summarize(samples, self.policy.stability_ratio)
        result = replace(
            result,
            batch_size=batch_size,
            gpu_time_ms=sum(window.gpu_time_ms for window in windows),
        )
        if self.policy.windows > 1:
            medians = tuple(window.p50_ms for window in windows)
            result = TimingStatistics(
                **{
                    **asdict(result),
                    "p50_ms": statistics.median(medians),
                    "batch_size": batch_size,
                    "gpu_time_ms": sum(window.gpu_time_ms for window in windows),
                    "window_medians_ms": medians,
                }
            )
        if not all(window.stable for window in windows):
            raise UnstableTimingError(result)
        return result

    def _measure_window(
        self,
        fn: Callable[..., Any],
        setup: Callable[[], tuple[Any, ...]],
        clear_cache: Callable[[], None],
        batch_size: int,
        validate: Callable[[tuple[Any, ...], Any], None] | None = None,
    ) -> TimingStatistics:
        samples: list[float] = []
        total_ms = 0.0
        while True:
            clear_cache()
            args = setup()
            output = None

            def run_block() -> None:
                nonlocal output
                for _ in range(batch_size):
                    output = fn(*args)

            block_elapsed = self.clock.measure(run_block)
            if validate is not None:
                validate(args, output)
            samples.append(block_elapsed / batch_size)
            total_ms += block_elapsed
            minimum_met = (
                len(samples) >= self.policy.sample_min_calls
                and total_ms >= self.policy.sample_min_ms
            )
            stats = summarize(samples, self.policy.stability_ratio)
            stats = replace(stats, batch_size=batch_size, gpu_time_ms=total_ms)
            if minimum_met and stats.stable:
                return stats
            if len(samples) >= self.policy.max_calls or total_ms >= self.policy.max_ms:
                if self.policy.stability_ratio is None:
                    return stats
                raise UnstableTimingError(stats)


class TorchCacheController:
    """Cold-cache controller that touches twice the last-level cache size."""

    def __init__(self, cache_bytes: int, device: str = "cuda"):
        import torch

        self._buffer = torch.empty(
            max(2 * cache_bytes, 1), dtype=torch.int8, device=device
        )

    def clear(self) -> None:
        self._buffer.zero_()


__all__ = [
    "AdaptiveTimer",
    "TimingPolicy",
    "TimingStatistics",
    "TorchCacheController",
    "TorchEventClock",
    "UnstableTimingError",
    "summarize",
]
