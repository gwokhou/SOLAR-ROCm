# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""One-way reader and AMD compatibility audit for SOL-ExecBench problems.

This module intentionally does not import the upstream ``sol_execbench``
package.  The official JSON/JSONL files are treated as an external data
protocol, keeping CUDA-only runtime dependencies out of SOLAR-ROCm.
"""

# The auditor deliberately has one exhaustive, ordered decision tree: every
# rejection must preserve its exact stage and evidence instead of falling back.
# pylint: disable=too-few-public-methods,too-many-locals,import-outside-toplevel,too-many-return-statements,too-many-branches,too-many-statements,missing-class-docstring,missing-function-docstring,exec-used,too-many-arguments,line-too-long

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import operator
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from solar.rocm import RocmEnvironment

SOL_EXECBENCH_SCHEMA_COMMIT = "a9fa0804c793d438e70850c33fe34426e66d53dd"
_FORBIDDEN_CYCLE_IMPORTS = frozenset({"solar", "sol_execbench"})
_CUDA_ONLY_IMPORTS = frozenset({"cuda", "cutlass", "cutile", "cudnn", "nvidia", "cupy"})
_DTYPE_BYTES = {
    "float64": 8.0,
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "float8_e4m3fn": 1.0,
    "float8_e5m2": 1.0,
    "float4_e2m1": 0.5,
    "float4_e2m1fn_x2": 0.5,
    "int64": 8.0,
    "int32": 4.0,
    "int16": 2.0,
    "int8": 1.0,
    "bool": 1.0,
}


class SolExecBenchFormatError(ValueError):
    """The official problem files violate the pinned schema contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _imports(source: str) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(source, mode="exec")):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_expression(expression: str, values: dict[str, int]) -> int:
    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise SolExecBenchFormatError(f"unsupported axis expression: {expression!r}")

    result = evaluate(ast.parse(expression, mode="eval"))
    if int(result) != result or result < 0:
        raise SolExecBenchFormatError(
            f"axis expression must resolve to a non-negative integer: {expression!r}"
        )
    return int(result)


@dataclass(frozen=True)
class OfficialWorkload:
    raw: dict[str, Any]
    line_number: int

    @property
    def uuid(self) -> str:
        return str(self.raw["uuid"])


@dataclass(frozen=True)
class SolExecBenchProblem:
    root: Path
    definition_path: Path
    workload_path: Path
    definition: dict[str, Any]
    workloads: tuple[OfficialWorkload, ...]
    blob_roots: tuple[Path, ...] = ()

    @classmethod
    def load(
        cls, root: str | Path, *, blob_roots: Iterable[str | Path] = ()
    ) -> "SolExecBenchProblem":
        problem_root = Path(root).resolve()
        definition_path = problem_root / "definition.json"
        workload_path = problem_root / "workload.jsonl"
        if not definition_path.is_file() or not workload_path.is_file():
            raise SolExecBenchFormatError(
                "problem directory must contain definition.json and workload.jsonl"
            )
        definition = json.loads(definition_path.read_text())
        if not isinstance(definition, dict):
            raise SolExecBenchFormatError("definition.json must be an object")
        required = {"name", "axes", "inputs", "outputs", "reference"}
        if not required.issubset(definition):
            raise SolExecBenchFormatError(
                "definition is missing: "
                + ", ".join(sorted(required - set(definition)))
            )
        tree = ast.parse(str(definition["reference"]), mode="exec")
        run_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "run"
            ),
            None,
        )
        if run_node is None:
            raise SolExecBenchFormatError("reference must define top-level run")
        parameters = [
            argument.arg
            for argument in [*run_node.args.posonlyargs, *run_node.args.args]
        ]
        if parameters != list(definition["inputs"]):
            raise SolExecBenchFormatError(
                "reference run parameters must exactly match definition input order"
            )
        custom = definition.get("custom_inputs_entrypoint")
        if custom and not any(
            isinstance(node, ast.FunctionDef) and node.name == custom
            for node in tree.body
        ):
            raise SolExecBenchFormatError("custom_inputs_entrypoint is not defined")

        workloads: list[OfficialWorkload] = []
        seen: set[str] = set()
        for line_number, line in enumerate(workload_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or not {"uuid", "axes", "inputs"}.issubset(
                raw
            ):
                raise SolExecBenchFormatError(f"invalid workload at line {line_number}")
            uuid = str(raw["uuid"])
            if not uuid or uuid in seen:
                raise SolExecBenchFormatError(
                    f"duplicate/empty workload UUID at line {line_number}"
                )
            seen.add(uuid)
            input_types = {
                str(spec.get("type", "random")) for spec in raw["inputs"].values()
            }
            if "custom" in input_types and input_types != {"custom"}:
                raise SolExecBenchFormatError(
                    "custom and non-custom inputs cannot be mixed"
                )
            workloads.append(OfficialWorkload(raw=raw, line_number=line_number))
        if not workloads:
            raise SolExecBenchFormatError("workload.jsonl has no workloads")
        resolved_blob_roots = tuple(Path(path).resolve() for path in blob_roots)
        return cls(
            problem_root,
            definition_path,
            workload_path,
            definition,
            tuple(workloads),
            resolved_blob_roots,
        )

    def resolve_blob(self, value: str) -> Path:
        """Resolve one official external input without searching ambient paths."""
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise SolExecBenchFormatError(
                f"safetensors path must be a safe relative path: {value!r}"
            )
        candidates = [self.root / relative]
        candidates.extend(root / relative for root in self.blob_roots)
        matches = [
            candidate.resolve() for candidate in candidates if candidate.is_file()
        ]
        if not matches:
            raise FileNotFoundError(
                f"missing external input {relative}; checked problem root and "
                f"{len(self.blob_roots)} explicit blob root(s)"
            )
        digests = {_sha256(path) for path in matches}
        if len(digests) != 1:
            raise SolExecBenchFormatError(
                f"ambiguous external input {relative}: explicit roots contain "
                "different files"
            )
        return matches[0]

    def resolved_axes(self, workload: OfficialWorkload) -> dict[str, int]:
        values: dict[str, int] = {}
        supplied = {str(key): int(value) for key, value in workload.raw["axes"].items()}
        for name, spec in self.definition["axes"].items():
            axis_type = str(spec.get("type", ""))
            if axis_type == "const":
                values[name] = int(spec["value"])
            elif axis_type == "var":
                if name not in supplied:
                    raise SolExecBenchFormatError(
                        f"workload {workload.uuid} lacks axis {name}"
                    )
                values[name] = supplied[name]
        pending = {
            name: str(spec["expression"])
            for name, spec in self.definition["axes"].items()
            if str(spec.get("type", "")) == "expr"
        }
        while pending:
            progressed = False
            for name, expression in list(pending.items()):
                referenced = {
                    node.id
                    for node in ast.walk(ast.parse(expression, mode="eval"))
                    if isinstance(node, ast.Name)
                }
                if referenced.issubset(values):
                    values[name] = _evaluate_expression(expression, values)
                    del pending[name]
                    progressed = True
            if not progressed:
                raise SolExecBenchFormatError(
                    "unresolvable/cyclic axis expressions: "
                    + ", ".join(sorted(pending))
                )
        return values

    def tensor_shape(
        self, spec: dict[str, Any], axes: dict[str, int]
    ) -> tuple[int, ...] | None:
        shape = spec.get("shape")
        if shape is None:
            return None
        result: list[int] = []
        for item in shape:
            text = str(item)
            if text.isdigit():
                result.append(int(text))
            elif text in axes:
                result.append(axes[text])
            else:
                result.append(_evaluate_expression(text, axes))
        return tuple(result)

    def reference_namespace(self) -> dict[str, Any]:
        namespace: dict[str, Any] = {"__name__": "_solar_official_reference"}
        exec(
            compile(
                str(self.definition["reference"]), str(self.definition_path), "exec"
            ),
            namespace,
        )
        return namespace

    def generate_inputs(
        self, workload: OfficialWorkload, device: str, *, seed: int = 200
    ) -> tuple[Any, ...]:
        import torch

        torch.manual_seed(seed)
        axes = self.resolved_axes(workload)
        specs = workload.raw["inputs"]
        namespace: dict[str, Any] | None = None
        custom_values: dict[str, Any] | None = None
        if {str(item.get("type", "random")) for item in specs.values()} == {"custom"}:
            namespace = self.reference_namespace()
            entrypoint = str(self.definition.get("custom_inputs_entrypoint") or "")
            if not entrypoint:
                raise SolExecBenchFormatError("custom workload has no input entrypoint")
            scalars = {
                name: item["value"]
                for name, item in specs.items()
                if str(item.get("type")) == "scalar"
            }
            custom_values = namespace[entrypoint](
                {**axes, **scalars}, torch.device(device)
            )

        dtype_map = {
            name: getattr(torch, name) for name in _DTYPE_BYTES if hasattr(torch, name)
        }
        values: list[Any] = []
        for name, tensor_spec in self.definition["inputs"].items():
            input_spec = specs.get(name, {"type": "random"})
            input_type = str(input_spec.get("type", "random"))
            if input_type == "scalar":
                values.append(input_spec["value"])
                continue
            if input_type == "custom":
                if custom_values is None or name not in custom_values:
                    raise SolExecBenchFormatError(
                        f"custom input factory omitted {name}"
                    )
                values.append(custom_values[name])
                continue
            dtype_name = str(tensor_spec["dtype"])
            if dtype_name not in dtype_map:
                raise RuntimeError(f"unsupported dtype {dtype_name}")
            dtype = dtype_map[dtype_name]
            shape = self.tensor_shape(tensor_spec, axes) or ()
            if input_type == "safetensors":
                import safetensors.torch

                candidate = self.resolve_blob(str(input_spec["path"]))
                tensor = safetensors.torch.load_file(str(candidate), device=device)[
                    str(input_spec["tensor_key"])
                ]
                values.append(tensor)
            elif dtype.is_floating_point:
                base = torch.randn(shape, dtype=torch.float32, device=device)
                values.append(base.to(dtype))
            elif dtype == torch.bool:
                values.append(
                    torch.randint(0, 2, shape, dtype=torch.bool, device=device)
                )
            else:
                values.append(torch.randint(-8, 8, shape, dtype=dtype, device=device))
        return tuple(values)


class AmdCompatibilityAuditor:
    """Produce evidence, never substitutions, for one AMD workload."""

    def __init__(self, problem: SolExecBenchProblem, *, device: str = "cuda"):
        self.problem = problem
        self.device = device
        self.environment = RocmEnvironment.detect()

    def _result(
        self,
        workload: OfficialWorkload,
        status: str,
        reason_code: str,
        *,
        stage: str,
        evidence: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "status": status,
            "reason_code": reason_code,
            "stage": stage,
            "problem": self.problem.definition["name"],
            "workload_uuid": workload.uuid,
            "definition_sha256": _sha256(self.problem.definition_path),
            "workload_jsonl_sha256": _sha256(self.problem.workload_path),
            "upstream_schema_commit": SOL_EXECBENCH_SCHEMA_COMMIT,
            "environment": self.environment.to_dict(),
            "evidence": evidence or {},
            "fallbacks_used": [],
        }
        if error is not None:
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        return result

    def audit(
        self, workload: OfficialWorkload, *, execute: bool = True
    ) -> dict[str, Any]:
        """Audit one official workload without substituting unsupported work."""
        imports = _imports(str(self.problem.definition["reference"]))
        cycles = sorted(imports & _FORBIDDEN_CYCLE_IMPORTS)
        if cycles:
            return self._result(
                workload,
                "incompatible",
                "cyclic_reference_dependency",
                stage="reference_imports",
                evidence={"imports": cycles},
            )
        cuda_only = sorted(imports & _CUDA_ONLY_IMPORTS)
        if cuda_only:
            return self._result(
                workload,
                "incompatible",
                "cuda_only_dependency",
                stage="reference_imports",
                evidence={"imports": cuda_only},
            )
        missing = sorted(
            name
            for name in imports
            if name not in {"__future__"} and importlib.util.find_spec(name) is None
        )
        if missing:
            return self._result(
                workload,
                "incompatible",
                "missing_amd_equivalent_library",
                stage="reference_imports",
                evidence={"imports": missing},
            )
        if not self.environment.supported_target:
            return self._result(
                workload,
                "not_checked",
                "toolchain_unavailable",
                stage="device_discovery",
            )

        blob_evidence: dict[str, Any] = {}
        for name, input_spec in workload.raw["inputs"].items():
            if str(input_spec.get("type")) != "safetensors":
                continue
            try:
                resolved = self.problem.resolve_blob(str(input_spec["path"]))
            except (FileNotFoundError, SolExecBenchFormatError) as exc:
                return self._result(
                    workload,
                    "not_checked",
                    "missing_external_input",
                    stage="external_input_resolution",
                    evidence={"input": name, "declared_path": input_spec.get("path")},
                    error=exc,
                )
            blob_evidence[name] = {
                "declared_path": str(input_spec["path"]),
                "resolved_path": str(resolved),
                "sha256": _sha256(resolved),
            }

        try:
            axes = self.problem.resolved_axes(workload)
            storage = 0.0
            dtypes: set[str] = set()
            tensors: dict[str, Any] = {}
            for side in ("inputs", "outputs"):
                for name, spec in self.problem.definition[side].items():
                    dtype = str(spec["dtype"])
                    dtypes.add(dtype)
                    if dtype not in _DTYPE_BYTES:
                        return self._result(
                            workload,
                            "incompatible",
                            "unsupported_dtype",
                            stage="static_dtype",
                            evidence={"dtype": dtype, "tensor": name},
                        )
                    shape = self.problem.tensor_shape(spec, axes)
                    elements = math.prod(shape or ()) if shape is not None else 1
                    byte_count = math.ceil(elements * _DTYPE_BYTES[dtype])
                    storage += byte_count
                    tensors[name] = {
                        "side": side,
                        "shape": shape,
                        "dtype": dtype,
                        "bytes": byte_count,
                    }
            total_memory = self.environment.total_memory_bytes
            if total_memory is not None and storage > total_memory:
                return self._result(
                    workload,
                    "incompatible",
                    "insufficient_device_capacity",
                    stage="static_capacity",
                    evidence={
                        "minimum_storage_bytes": int(storage),
                        "device_total_bytes": total_memory,
                        "tensors": tensors,
                    },
                )
            if not execute:
                return self._result(
                    workload,
                    "compatible",
                    "compatible",
                    stage="static_complete",
                    evidence={
                        "minimum_storage_bytes": int(storage),
                        "dtypes": sorted(dtypes),
                        "external_inputs": blob_evidence,
                    },
                )

            import torch

            selected = torch.device(self.device)
            if selected.type != "cuda" or not getattr(torch.version, "hip", None):
                return self._result(
                    workload,
                    "not_checked",
                    "selected_device_is_not_amd_rocm",
                    stage="selected_device_validation",
                    evidence={"requested_device": self.device},
                )
            selected_index = (
                selected.index
                if selected.index is not None
                else torch.cuda.current_device()
            )
            if selected_index < 0 or selected_index >= torch.cuda.device_count():
                return self._result(
                    workload,
                    "not_checked",
                    "selected_device_unavailable",
                    stage="selected_device_validation",
                    evidence={"requested_device": self.device},
                )
            properties = torch.cuda.get_device_properties(selected_index)
            selected_gfx = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
            if not selected_gfx.startswith("gfx"):
                return self._result(
                    workload,
                    "not_checked",
                    "selected_device_is_not_amd_rocm",
                    stage="selected_device_validation",
                    evidence={
                        "requested_device": self.device,
                        "device_name": str(properties.name),
                        "gfx_target": selected_gfx or None,
                    },
                )
            selected_device = {
                "requested_device": self.device,
                "resolved_device": f"cuda:{selected_index}",
                "device_name": str(properties.name),
                "gfx_target": selected_gfx,
                "hip_version": str(torch.version.hip),
            }

            for dtype_name in sorted(dtypes):
                dtype = getattr(torch, dtype_name, None)
                if not isinstance(dtype, torch.dtype):
                    return self._result(
                        workload,
                        "incompatible",
                        "unsupported_dtype",
                        stage="device_dtype_probe",
                        evidence={"dtype": dtype_name},
                    )
                try:
                    torch.empty((1,), dtype=dtype, device=self.device).clone()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    reason = (
                        "unsupported_quantization_format"
                        if "float8" in dtype_name or "float4" in dtype_name
                        else "unsupported_dtype"
                    )
                    return self._result(
                        workload,
                        "incompatible",
                        reason,
                        stage="device_dtype_probe",
                        evidence={"dtype": dtype_name},
                        error=exc,
                    )
            try:
                inputs = self.problem.generate_inputs(workload, self.device)
            except torch.OutOfMemoryError as exc:
                free, total = torch.cuda.mem_get_info()
                return self._result(
                    workload,
                    "execution_failed",
                    "runtime_oom",
                    stage="input_generation",
                    evidence={"free_bytes": free, "total_bytes": total},
                    error=exc,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                message = str(exc).lower()
                reason = (
                    "unsupported_quantization_format"
                    if "float4" in message or "float8" in message
                    else "input_generation_failed"
                )
                status = (
                    "incompatible"
                    if reason.startswith("unsupported")
                    else "execution_failed"
                )
                return self._result(
                    workload, status, reason, stage="input_generation", error=exc
                )
            try:
                namespace = self.problem.reference_namespace()
                outputs = namespace["run"](*inputs)
                torch.cuda.synchronize()
                expected_specs = list(self.problem.definition["outputs"].items())
                if isinstance(outputs, dict):
                    actual_outputs = [outputs[name] for name, _ in expected_specs]
                elif isinstance(outputs, (tuple, list)):
                    actual_outputs = list(outputs)
                else:
                    actual_outputs = [outputs]
                if len(actual_outputs) != len(expected_specs):
                    raise RuntimeError(
                        "reference output arity disagrees with definition"
                    )
                axes = self.problem.resolved_axes(workload)
                for actual, (name, spec) in zip(actual_outputs, expected_specs):
                    expected_shape = self.problem.tensor_shape(spec, axes)
                    if not isinstance(actual, torch.Tensor):
                        raise RuntimeError(f"reference output {name} is not a tensor")
                    if (
                        expected_shape is not None
                        and tuple(actual.shape) != expected_shape
                    ):
                        raise RuntimeError(
                            f"reference output {name} shape {tuple(actual.shape)} != {expected_shape}"
                        )
                    if str(actual.dtype).replace("torch.", "") != str(spec["dtype"]):
                        raise RuntimeError(
                            f"reference output {name} dtype {actual.dtype} != {spec['dtype']}"
                        )
            except torch.OutOfMemoryError as exc:
                free, total = torch.cuda.mem_get_info()
                return self._result(
                    workload,
                    "execution_failed",
                    "runtime_oom",
                    stage="reference_execution",
                    evidence={"free_bytes": free, "total_bytes": total},
                    error=exc,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                message = str(exc).lower()
                unsupported = any(
                    token in message
                    for token in (
                        "not implemented",
                        "unsupported",
                        "hip error",
                        "invalid device function",
                    )
                )
                return self._result(
                    workload,
                    "incompatible" if unsupported else "execution_failed",
                    (
                        "unsupported_operator"
                        if unsupported
                        else "reference_execution_failed"
                    ),
                    stage="reference_execution",
                    error=exc,
                )
            del outputs, inputs
            return self._result(
                workload,
                "compatible",
                "compatible",
                stage="reference_complete",
                evidence={
                    "minimum_storage_bytes": int(storage),
                    "dtypes": sorted(dtypes),
                    "external_inputs": blob_evidence,
                    "selected_device": selected_device,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._result(
                workload,
                "execution_failed",
                "reference_execution_failed",
                stage="audit_internal",
                error=exc,
            )


def write_compatibility_artifact(result: dict[str, Any], path: str | Path) -> str:
    """Write one hashable compatibility evidence record."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return _sha256(output)


def standalone_reference_source(problem: SolExecBenchProblem) -> str:
    """Materialize a replayable input factory without importing SOLAR/upstream."""
    workload_data = {workload.uuid: workload.raw for workload in problem.workloads}
    shape_data = {
        workload.uuid: {
            name: problem.tensor_shape(spec, problem.resolved_axes(workload))
            for name, spec in problem.definition["inputs"].items()
        }
        for workload in problem.workloads
    }
    axes_data = {
        workload.uuid: problem.resolved_axes(workload) for workload in problem.workloads
    }
    definition_inputs = problem.definition["inputs"]
    custom = problem.definition.get("custom_inputs_entrypoint")
    prelude = f"""\n# Generated from pinned SOL-ExecBench JSON; no SOLAR runtime dependency.\nimport json as _json\nfrom pathlib import Path as _Path\nimport torch as _torch\n_WORKLOADS = _json.loads({json.dumps(workload_data)!r})\n_SHAPES = _json.loads({json.dumps(shape_data)!r})\n_AXES = _json.loads({json.dumps(axes_data)!r})\n_INPUTS = _json.loads({json.dumps(definition_inputs)!r})\n_CUSTOM_ENTRYPOINT = {custom!r}\n_ROOT = _Path(__file__).resolve().parent\n\ndef get_inputs(parameters, device):\n    uuid = str(parameters["uuid"])\n    workload = _WORKLOADS[uuid]\n    seed = int(parameters.get("seed", 200))\n    _torch.manual_seed(seed)\n    specs = workload["inputs"]\n    if specs and {{str(item.get("type", "random")) for item in specs.values()}} == {{"custom"}}:\n        custom_values = globals()[_CUSTOM_ENTRYPOINT](dict(_AXES[uuid]), _torch.device(device))\n    else:\n        custom_values = None\n    values = []\n    for name, tensor_spec in _INPUTS.items():\n        input_spec = specs.get(name, {{"type": "random"}})\n        input_type = str(input_spec.get("type", "random"))\n        if input_type == "scalar":\n            values.append(input_spec["value"])\n            continue\n        if input_type == "custom":\n            values.append(custom_values[name])\n            continue\n        dtype = getattr(_torch, str(tensor_spec["dtype"]))\n        shape = tuple(_SHAPES[uuid][name] or ())\n        if input_type == "safetensors":\n            import safetensors.torch as _st\n            values.append(_st.load_file(str(_ROOT / "data" / input_spec["path"]), device=str(device))[input_spec["tensor_key"]])\n        elif dtype.is_floating_point:\n            values.append(_torch.randn(shape, dtype=_torch.float32, device=device).to(dtype))\n        elif dtype == _torch.bool:\n            values.append(_torch.randint(0, 2, shape, dtype=_torch.bool, device=device))\n        else:\n            values.append(_torch.randint(-8, 8, shape, dtype=dtype, device=device))\n    return tuple(values)\n"""
    return str(problem.definition["reference"]).rstrip() + "\n" + prelude


__all__ = [
    "AmdCompatibilityAuditor",
    "OfficialWorkload",
    "SOL_EXECBENCH_SCHEMA_COMMIT",
    "SolExecBenchFormatError",
    "SolExecBenchProblem",
    "write_compatibility_artifact",
    "standalone_reference_source",
]
