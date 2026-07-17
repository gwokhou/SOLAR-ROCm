# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ROCm evaluator for SOLAR YAML benchmark packages."""

from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from solar.benchmark.backends import BackendUnavailable, get_backend
from solar.benchmark.clock_lock import (
    ClockLockLease,
    acquire_clock_lock,
    query_clock_level,
)
from solar.benchmark.models import (
    BaselineRegistry,
    BenchmarkSpec,
    SolutionSpec,
    WorkloadSpec,
    canonical_hash,
)
from solar.benchmark.scoring import calculate_sol_score
from solar.benchmark.staging import stage_solution
from solar.benchmark.timing import (
    AdaptiveTimer,
    TimingPolicy,
    TorchCacheController,
    UnstableTimingError,
)
from solar.rocm import ArchitectureProfile, RocmEnvironment


@dataclass
class WorkloadEvaluation:
    name: str
    status: str
    workload_hash: str | None = None
    correct: bool = False
    theoretical_solar_ms: float | None = None
    calibrated_solar_ms: float | None = None
    candidate_latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    sol_score: float | None = None
    timing: dict[str, Any] | None = None
    analysis_sha256: str | None = None
    source_graph_sha256: str | None = None
    failure: str | None = None


@dataclass
class EvaluationReport:
    schema_version: int
    benchmark_name: str
    benchmark_hash: str
    reference_sha256: str
    solution_name: str
    solution_hash: str
    backend: str
    timing_profile: str
    cache_policy: str
    environment: dict[str, Any]
    environment_hash: str
    architecture: dict[str, Any]
    architecture_hash: str
    clocks_locked: bool
    publishable: bool
    workloads: list[WorkloadEvaluation] = field(default_factory=list)
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _clone(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.clone()
    except ImportError:
        pass
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return value


def _equal(actual: Any, expected: Any, atol: float, rtol: float) -> bool:
    try:
        import torch

        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            return (
                actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and bool(
                    torch.allclose(
                        actual, expected, atol=atol, rtol=rtol, equal_nan=True
                    )
                )
            )
    except ImportError:
        pass
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(
            _equal(a, e, atol, rtol) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equal(actual[key], expected[key], atol, rtol) for key in actual
        )
    return actual == expected


class _TimedCorrectnessError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


class _RuntimeIntegrityGuard:
    """Detect persistent monkey patches and evaluator-visible worker threads."""

    def __init__(self) -> None:
        import torch

        self._symbols = {
            "torch.allclose": (torch, "allclose", torch.allclose),
            "torch.cuda.Event": (torch.cuda, "Event", torch.cuda.Event),
            "torch.cuda.synchronize": (
                torch.cuda,
                "synchronize",
                torch.cuda.synchronize,
            ),
            "torch.Tensor.clone": (torch.Tensor, "clone", torch.Tensor.clone),
            "torch.Tensor.data_ptr": (
                torch.Tensor,
                "data_ptr",
                torch.Tensor.data_ptr,
            ),
        }
        self._thread_ids = {thread.ident for thread in threading.enumerate()}

    def check(self) -> None:
        for name, (owner, attribute, expected) in self._symbols.items():
            if getattr(owner, attribute) is not expected:
                raise _TimedCorrectnessError(
                    "reward_hack", f"candidate modified evaluator primitive {name}"
                )
        unexpected = [
            thread.name
            for thread in threading.enumerate()
            if thread.ident not in self._thread_ids and thread.is_alive()
        ]
        if unexpected:
            raise _TimedCorrectnessError(
                "reward_hack",
                "candidate created evaluator-visible worker threads: "
                + ", ".join(unexpected),
            )


class _ShiftingInputPool:
    """Materialize each call at a never-repeated, aligned tensor address."""

    def __init__(self, factory: Callable[[], tuple[Any, ...]], max_calls: int) -> None:
        self._factory = factory
        self._max_calls = max_calls
        self._index = 0
        self._bases: dict[
            tuple[Any, ...], tuple[Any, tuple[int, ...], tuple[int, ...], int]
        ] = {}
        self._pending: dict[int, Any] = {}
        self._seen_ptrs: set[int] = set()

    def _shift(self, value: Any, path: tuple[Any, ...]) -> Any:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                shape = tuple(value.shape)
                stride = tuple(value.stride())
                span = (
                    1 + sum((size - 1) * step for size, step in zip(shape, stride))
                    if value.numel()
                    else 0
                )
                alignment = max(1, 64 // max(value.element_size(), 1))
                if path not in self._bases:
                    base = torch.empty(
                        span + self._max_calls * alignment,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    self._bases[path] = (base, shape, stride, alignment)
                base, expected_shape, expected_stride, alignment = self._bases[path]
                if shape != expected_shape or stride != expected_stride:
                    raise ValueError("input factory changed tensor shape or layout")
                shifted = torch.as_strided(
                    base,
                    size=shape,
                    stride=stride,
                    storage_offset=self._index * alignment,
                )
                shifted.copy_(value)
                pointer = shifted.data_ptr()
                if value.numel() and pointer in self._seen_ptrs:
                    raise RuntimeError("input tensor address was reused during timing")
                if value.numel():
                    self._seen_ptrs.add(pointer)
                return shifted
        except ImportError:
            pass
        if isinstance(value, tuple):
            return tuple(
                self._shift(item, (*path, index)) for index, item in enumerate(value)
            )
        if isinstance(value, list):
            return [
                self._shift(item, (*path, index)) for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            return {key: self._shift(item, (*path, key)) for key, item in value.items()}
        return value

    def next(self) -> tuple[Any, ...]:
        if self._index >= self._max_calls:
            raise RuntimeError("unique input-address budget exhausted")
        source = tuple(self._factory())
        pristine = _clone(source)
        shifted = self._shift(source, ())
        self._pending[id(shifted)] = pristine
        self._index += 1
        return shifted

    def take_pristine(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        try:
            return self._pending.pop(id(args))
        except KeyError as exc:
            raise RuntimeError("timing input provenance was lost") from exc


class RocmEvaluator:
    """Evaluate one candidate with correctness gating and optional SOL scoring."""

    def __init__(
        self,
        architecture: ArchitectureProfile | None = None,
        environment: RocmEnvironment | None = None,
    ):
        self.architecture = architecture or ArchitectureProfile.load("RX_9060_XT")
        if self.architecture.vendor.upper() != "AMD":
            raise ValueError("ROCm executable evaluation requires an AMD profile")
        self.environment = environment

    def evaluate(
        self,
        benchmark: BenchmarkSpec | str | Path,
        solution: SolutionSpec | str | Path,
        *,
        baseline: BaselineRegistry | str | Path | None = None,
        timing_profile: str = "standard",
        lock_clocks: bool = True,
        trusted_local: bool = True,
        calibrated_solar_ms: dict[str, float] | None = None,
    ) -> EvaluationReport:
        if not trusted_local:
            raise ValueError(
                "untrusted evaluations must be launched through the ROCm Docker runner"
            )
        benchmark = (
            benchmark
            if isinstance(benchmark, BenchmarkSpec)
            else BenchmarkSpec.load(benchmark)
        )
        solution = (
            solution
            if isinstance(solution, SolutionSpec)
            else SolutionSpec.load(solution)
        )
        baseline = (
            baseline
            if isinstance(baseline, BaselineRegistry) or baseline is None
            else BaselineRegistry.load(baseline)
        )
        policy = TimingPolicy.for_name(timing_profile)
        environment = self.environment or RocmEnvironment.detect()
        environment_data = environment.to_dict()
        environment_hash = canonical_hash(environment_data)
        architecture_data = self.architecture.to_dict()
        architecture_identity = dict(architecture_data)
        architecture_identity.pop("source", None)
        report = EvaluationReport(
            schema_version=1,
            benchmark_name=benchmark.name,
            benchmark_hash=benchmark.raw_hash,
            reference_sha256=benchmark.reference_sha256,
            solution_name=solution.name,
            solution_hash=solution.raw_hash,
            backend=solution.backend,
            timing_profile=policy.name,
            cache_policy=benchmark.cache_policy,
            environment=environment_data,
            environment_hash=environment_hash,
            architecture=architecture_data,
            architecture_hash=canonical_hash(architecture_identity),
            clocks_locked=False,
            publishable=False,
        )
        if not environment.supported_target:
            report.failure = f"ROCm GPU target unavailable: {environment.gfx_target}"
            return report
        if environment.gfx_target != self.architecture.gfx_target:
            report.failure = (
                f"architecture target mismatch: detected {environment.gfx_target}; "
                f"profile requires {self.architecture.gfx_target}"
            )
            return report
        if environment.gfx_target not in solution.gfx_targets:
            report.failure = (
                f"solution does not declare detected target {environment.gfx_target}"
            )
            return report

        if lock_clocks:
            lease = acquire_clock_lock()
        else:
            observed = query_clock_level()
            externally_locked = bool(observed) and all(
                "STABLE_PEAK" in level.upper() for level in observed
            )
            lease = ClockLockLease(externally_locked, False)
        try:
            report.clocks_locked = lease.locked
            baseline_error = None
            if baseline is not None:
                try:
                    baseline.assert_compatible(
                        benchmark,
                        environment_hash,
                        report.architecture_hash,
                        policy.name,
                        environment.gfx_target or "",
                        lease.locked,
                    )
                except ValueError as exc:
                    baseline_error = str(exc)
            backend = get_backend(solution.backend)
            try:
                backend.assert_available(environment)
            except BackendUnavailable as exc:
                report.failure = str(exc)
                return report
            with stage_solution(solution) as staged:
                reference_module = _load_module(
                    benchmark.source_root / benchmark.reference_source,
                    f"_solar_reference_{benchmark.raw_hash[:12]}",
                )
                reference = getattr(reference_module, benchmark.reference_entry_point)
                input_factory = getattr(reference_module, benchmark.input_factory)
                integrity = _RuntimeIntegrityGuard()
                candidate = backend.load(
                    solution, staged.root, environment.gfx_target or ""
                )
                integrity.check()
                cache = (
                    TorchCacheController(self.architecture.cache_flush_bytes)
                    if benchmark.cache_policy == "cold"
                    else None
                )
                for workload in benchmark.workloads:
                    report.workloads.append(
                        self._evaluate_workload(
                            workload,
                            benchmark,
                            candidate,
                            reference,
                            input_factory,
                            policy,
                            cache,
                            baseline,
                            baseline_error,
                            calibrated_solar_ms or {},
                            lease.locked,
                            integrity,
                        )
                    )
            report.publishable = bool(
                policy.publishable
                and lease.locked
                and baseline is not None
                and baseline_error is None
                and report.workloads
                and all(
                    item.status == "passed" and item.sol_score is not None
                    for item in report.workloads
                )
            )
            return report
        except Exception as exc:
            report.failure = f"infrastructure_error: {type(exc).__name__}: {exc}"
            return report
        finally:
            lease.release()

    def _evaluate_workload(
        self,
        workload: WorkloadSpec,
        benchmark: BenchmarkSpec,
        candidate: Callable[..., Any],
        reference: Callable[..., Any],
        input_factory: Callable[..., Any],
        policy: TimingPolicy,
        cache: TorchCacheController | None,
        baseline: BaselineRegistry | None,
        baseline_error: str | None,
        calibrated: dict[str, float],
        clocks_locked: bool,
        integrity: _RuntimeIntegrityGuard,
    ) -> WorkloadEvaluation:
        analysis = workload.analysis
        if analysis is None:
            return WorkloadEvaluation(
                name=workload.name,
                status="invalid_analysis",
                failure="workload has no bound SOLAR analysis artifact",
            )
        item = WorkloadEvaluation(
            name=workload.name,
            status="invalid",
            workload_hash=canonical_hash(
                {
                    "name": workload.name,
                    "parameters": workload.parameters,
                    "analysis_sha256": analysis.sha256,
                }
            ),
            analysis_sha256=analysis.sha256,
            source_graph_sha256=analysis.source_graph_sha256,
        )
        try:

            def guarded_candidate(*args: Any) -> Any:
                integrity.check()
                result = candidate(*args)
                integrity.check()
                return result

            for seed in range(200, 210):
                parameters = {**workload.parameters, "seed": seed}
                inputs = tuple(input_factory(parameters, "cuda"))
                candidate_inputs = _clone(inputs)
                before = _clone(candidate_inputs)
                expected = reference(*_clone(inputs))
                actual = guarded_candidate(*candidate_inputs)
                if not _equal(candidate_inputs, before, benchmark.atol, benchmark.rtol):
                    item.status = "reward_hack"
                    item.failure = "candidate modified benchmark inputs"
                    return item
                if not _equal(actual, expected, benchmark.atol, benchmark.rtol):
                    item.status = "incorrect"
                    item.failure = (
                        f"candidate output differs from reference for seed {seed}"
                    )
                    item.sol_score = 0.0 if baseline is not None else None
                    return item
            theoretical_ms = (
                self.architecture.theoretical_seconds_by_precision(
                    analysis.macs_by_precision, analysis.fused_bytes
                )
                * 1000.0
            )
            item.theoretical_solar_ms = theoretical_ms
            item.calibrated_solar_ms = calibrated.get(workload.name)
            timer = AdaptiveTimer(policy)
            seed = 10_000

            def make_inputs() -> tuple[Any, ...]:
                nonlocal seed
                seed += 1
                return tuple(
                    input_factory({**workload.parameters, "seed": seed}, "cuda")
                )

            pool = _ShiftingInputPool(
                make_inputs,
                max_calls=policy.max_calls * (policy.windows + 1) + 32,
            )
            timed_call = 0

            def validate_timed(args: tuple[Any, ...], actual: Any) -> None:
                nonlocal timed_call
                timed_call += 1
                pristine = pool.take_pristine(args)
                if not _equal(args, pristine, benchmark.atol, benchmark.rtol):
                    raise _TimedCorrectnessError(
                        "reward_hack", "candidate modified timed benchmark inputs"
                    )
                expected = reference(*_clone(pristine))
                if not _equal(actual, expected, benchmark.atol, benchmark.rtol):
                    raise _TimedCorrectnessError(
                        "incorrect",
                        f"candidate output differs during timed call {timed_call}",
                    )

            stats = timer.measure(
                guarded_candidate,
                setup=pool.next,
                clear_cache=cache.clear if cache is not None else None,
                allow_batching=False,
                validate=validate_timed,
            )
            for seed in range(300, 310):
                parameters = {**workload.parameters, "seed": seed}
                inputs = tuple(input_factory(parameters, "cuda"))
                candidate_inputs = _clone(inputs)
                before = _clone(candidate_inputs)
                expected = reference(*_clone(inputs))
                actual = guarded_candidate(*candidate_inputs)
                if not _equal(candidate_inputs, before, benchmark.atol, benchmark.rtol):
                    raise _TimedCorrectnessError(
                        "reward_hack", "candidate modified post-timing inputs"
                    )
                if not _equal(actual, expected, benchmark.atol, benchmark.rtol):
                    raise _TimedCorrectnessError(
                        "incorrect",
                        f"candidate output differs after timing for seed {seed}",
                    )
            item.correct = True
            item.timing = stats.to_dict()
            item.candidate_latency_ms = stats.p50_ms
            if baseline is not None:
                item.baseline_latency_ms = baseline.workloads.get(workload.name)
            if baseline_error:
                item.failure = baseline_error
                item.status = "invalid_baseline"
            elif baseline is not None and item.baseline_latency_ms is None:
                item.failure = "baseline has no matching workload"
                item.status = "invalid_baseline"
            elif baseline is not None and policy.publishable and clocks_locked:
                assert item.baseline_latency_ms is not None
                item.sol_score = calculate_sol_score(
                    item.baseline_latency_ms, item.candidate_latency_ms, theoretical_ms
                )
                item.status = "passed"
            else:
                item.status = "diagnostic"
                if baseline is not None and not clocks_locked:
                    item.failure = (
                        "verified STABLE_PEAK clock lock is required for scoring"
                    )
                elif not policy.publishable:
                    item.failure = "quick timing is not publishable"
            return item
        except UnstableTimingError as exc:
            item.status = "unstable_timing"
            item.timing = exc.statistics.to_dict()
            item.candidate_latency_ms = exc.statistics.p50_ms
            item.failure = str(exc)
            return item
        except _TimedCorrectnessError as exc:
            item.status = exc.status
            item.correct = False
            item.sol_score = 0.0 if baseline is not None else None
            item.failure = str(exc)
            return item
        except Exception as exc:
            item.status = "runtime_error"
            item.failure = f"{type(exc).__name__}: {exc}"
            return item


__all__ = ["EvaluationReport", "RocmEvaluator", "WorkloadEvaluation"]
