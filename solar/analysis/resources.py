# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned AMD compute-resource accounting for executable einsum graphs.

The counters in this module are hardware independent.  Architecture profiles
map them to conservative, sourced upper rates.  Official analysis is
fail-closed: an executable operation is either classified here or rejected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from solar.common.constants import normalize_dtype

RESOURCE_MODEL_VERSION = "amd_resource_v1"

_VIEW_OPS = frozenset(
    {
        "detach",
        "expand",
        "flatten",
        "getitem",
        "identity",
        "narrow",
        "permute",
        "reshape",
        "select",
        "slice",
        "squeeze",
        "transpose",
        "unsqueeze",
        "view",
    }
)
_MEMORY_ONLY_OPS = frozenset(
    {
        "cat",
        "chunk",
        "clone",
        "contiguous",
        "copy",
        "copy_",
        "pad",
        "repeat",
        "repeat_interleave",
        "split",
        "stack",
        "tensor_split",
    }
)
_MFMA_OPS = frozenset(
    {
        "addmm",
        "bmm",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "linear",
        "matmul",
        "mm",
    }
)
_SFU_OPS = frozenset(
    {
        "cos",
        "exp",
        "log",
        "pow",
        "rsqrt",
        "sin",
        "sqrt",
        "tanh",
    }
)
_COMPOSITE_SFU_OPS = frozenset(
    {
        "elu",
        "gelu",
        "hardsigmoid",
        "hardswish",
        "mish",
        "sigmoid",
        "silu",
        "softplus",
    }
)
_REDUCTION_OPS = frozenset(
    {"amax", "amin", "argmax", "argmin", "logsumexp", "mean", "prod", "sum"}
)
_NORMALIZATION_OPS = frozenset(
    {"batch_norm", "group_norm", "layer_norm", "log_softmax", "softmax"}
)
_ATOMIC_OPS = frozenset(
    {
        "__setitem__",
        "index_add",
        "index_copy",
        "index_put",
        "scatter",
        "scatter_add",
    }
)
_SCAN_SORT_OPS = frozenset(
    {"argsort", "cummax", "cummin", "cumprod", "cumsum", "sort", "topk"}
)
_CONVERSION_OPS = frozenset(
    {
        "bfloat16",
        "dequantize",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_tensor_affine",
        "float",
        "half",
        "int",
        "long",
        "quantize_per_channel",
        "quantize_per_tensor",
        "to",
        "type",
        "type_as",
    }
)
_INDEX_OPS = frozenset(
    {"embedding", "embedding_bag", "gather", "index_select", "tril", "triu"}
)
_VALU_OPS = frozenset(
    {
        "abs",
        "add",
        "bitwise_and",
        "bitwise_not",
        "clamp",
        "div",
        "eq",
        "ge",
        "gt",
        "le",
        "lt",
        "maximum",
        "masked_fill",
        "minimum",
        "mul",
        "ne",
        "neg",
        "ones_like",
        "relu",
        "square",
        "sub",
        "where",
        "zeros_like",
    }
)


class ResourceClassificationError(ValueError):
    """Raised when a semantic compute node has no exact resource rule."""


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) & {"value", "dtype"}:
        return _unwrap(value.get("value", value.get("dtype")))
    if isinstance(value, (list, tuple)):
        return [_unwrap(item) for item in value]
    return value


def _elements(shapes: list[Any]) -> list[int]:
    result: list[int] = []
    for shape in shapes:
        if isinstance(shape, list):
            result.append(int(math.prod(int(dim) for dim in shape)))
        else:
            result.append(0)
    return result


def _mode(dtype: Any, fallback: str) -> str:
    normalized = normalize_dtype(dtype, fallback)
    if normalized.startswith(("int", "uint")):
        return "integer"
    return normalized


def _accumulation_mode(dtype: Any, fallback: str) -> str:
    source = normalize_dtype(dtype, fallback)
    if source in {"fp16", "bf16", "fp8", "nvfp4"}:
        return f"{source}->fp32"
    if source.startswith(("int", "uint")):
        return f"{source}->int32"
    return f"{source}->{source}"


def _reduction_groups(shape: list[int] | None, semantic: Mapping[str, Any]) -> int:
    if not shape:
        return 1
    kwargs = semantic.get("kwargs") or {}
    dim = _unwrap(kwargs.get("dim"))
    if dim is None:
        positional = [
            _unwrap(argument)
            for argument in (semantic.get("arguments") or [])
            if not (isinstance(argument, Mapping) and "tensor" in argument)
        ]
        if positional:
            dim = positional[0]
    if dim is None:
        return 1
    dims = [dim] if isinstance(dim, int) else list(dim)
    rank = len(shape)
    reduced = 1
    for item in dims:
        index = int(item) % rank
        reduced *= int(shape[index])
    return max(1, int(math.prod(shape)) // max(1, reduced))


def classify_layer_resources(
    layer: Mapping[str, Any],
    *,
    macs: int,
    fallback_precision: str,
    strict: bool,
    compute_precision: str | None = None,
) -> dict[str, Any]:
    """Return deterministic resource work for one executable graph layer."""
    semantic = layer.get("semantic_op") or {}
    target = str(semantic.get("target") or layer.get("type") or "").lower()
    target = target.rsplit(".", maxsplit=1)[-1]
    if target.endswith("_") and not target.endswith("__"):
        target = target[:-1]
    kind = str(semantic.get("kind", ""))
    shapes = layer.get("tensor_shapes") or {}
    input_shapes = list(shapes.get("inputs") or [])
    output_shapes = list(shapes.get("outputs") or [])
    input_elements = _elements(input_shapes)
    output_elements = _elements(output_shapes)
    inputs = list((layer.get("tensor_dtypes") or {}).get("inputs") or [])
    outputs = list((layer.get("tensor_dtypes") or {}).get("outputs") or [])
    dtype = compute_precision or (
        inputs[0] if inputs else (outputs[0] if outputs else fallback_precision)
    )
    mode = _mode(dtype, fallback_precision)
    output_n = max(output_elements, default=0)
    input_n = max(input_elements, default=0)
    work: dict[str, dict[str, int]] = defaultdict(dict)
    formulas: list[str] = []

    def add(resource: str, resource_mode: str, amount: int, formula: str) -> None:
        if amount <= 0:
            return
        work[resource][resource_mode] = work[resource].get(resource_mode, 0) + int(
            amount
        )
        formulas.append(formula)

    if kind == "einsum" or (target in _MFMA_OPS and macs > 0):
        add(
            "mfma",
            _accumulation_mode(dtype, fallback_precision),
            2 * int(macs),
            "2 * contraction_macs",
        )
    elif target == "scaled_dot_product_attention":
        q_shape = input_shapes[0] if input_shapes else []
        k_shape = input_shapes[1] if len(input_shapes) > 1 else []
        if len(q_shape) < 2 or len(k_shape) < 2:
            if strict:
                raise ResourceClassificationError(
                    "scaled_dot_product_attention requires ranked Q/K tensors"
                )
        else:
            q_rows = int(math.prod(q_shape[:-1]))
            q_width = int(q_shape[-1])
            k_rows_per_batch = int(k_shape[-2])
            score_elements = q_rows * k_rows_per_batch
            add(
                "mfma",
                _accumulation_mode(dtype, fallback_precision),
                4 * q_rows * k_rows_per_batch * q_width,
                "QK and probability-V contractions",
            )
            add(
                "reduction",
                mode,
                2 * max(0, score_elements - q_rows),
                "softmax max+sum combines",
            )
            add("sfu", mode, score_elements, "softmax exponentials")
            add("valu", mode, 2 * score_elements, "softmax subtract+divide")
    elif target in _VIEW_OPS:
        return {
            "model_version": RESOURCE_MODEL_VERSION,
            "work": {},
            "classification": "exempt",
            "exemption_reason": "metadata_or_alias_only",
            "formulas": [],
        }
    elif target in _MEMORY_ONLY_OPS:
        return {
            "model_version": RESOURCE_MODEL_VERSION,
            "work": {},
            "classification": "exempt",
            "exemption_reason": "memory_traffic_only",
            "formulas": [],
        }
    elif target in _ATOMIC_OPS or bool((semantic.get("effects") or {}).get("atomic")):
        # Indexed write APIs order tensor operands as destination, index, and
        # source/value.  Every source element is one potentially conflicting
        # update; counting only index elements undercounts vector rows.
        updates = (
            input_elements[-1]
            if len(input_elements) >= 2 and input_elements[-1] > 0
            else max(output_n, input_n)
        )
        update_dtype = inputs[-1] if len(inputs) >= 2 else dtype
        add(
            "atomic",
            _mode(update_dtype, fallback_precision),
            updates,
            "one atomic/conflicting update per source element",
        )
    elif target in _SCAN_SORT_OPS:
        add("scan_sort", mode, max(input_n, output_n), "one mandatory item visit")
    elif target in _CONVERSION_OPS:
        source_mode = _mode(inputs[0] if inputs else dtype, fallback_precision)
        destination_mode = _mode(outputs[0] if outputs else dtype, fallback_precision)
        if source_mode == destination_mode:
            return {
                "model_version": RESOURCE_MODEL_VERSION,
                "work": {},
                "classification": "exempt",
                "exemption_reason": "same_dtype_conversion_noop",
                "formulas": [],
            }
        add(
            "conversion",
            f"{source_mode}->{destination_mode}",
            output_n,
            "one conversion per output element",
        )
        if "quantize" in target or target == "dequantize":
            add("valu", destination_mode, 2 * output_n, "quantization scale and offset")
            if "per_channel" in target:
                add(
                    "reduction",
                    source_mode,
                    max(0, input_n - output_n),
                    "per-channel/block scale reduction",
                )
    elif target in _NORMALIZATION_OPS:
        shape = (
            output_shapes[0]
            if output_shapes and isinstance(output_shapes[0], list)
            else None
        )
        groups = _reduction_groups(shape, semantic)
        combines = max(0, output_n - groups)
        if target in {"softmax", "log_softmax"}:
            add("reduction", mode, 2 * combines, "maximum and sum reductions")
            add("sfu", mode, output_n, "exponential or logarithm")
            add("valu", mode, 2 * output_n, "normalization arithmetic")
        else:
            add("reduction", mode, 2 * combines, "mean and variance reductions")
            add("sfu", mode, groups, "inverse square root per group")
            add("valu", mode, 5 * output_n, "center, scale, normalize, affine")
    elif target in _REDUCTION_OPS:
        shape = (
            input_shapes[0]
            if input_shapes and isinstance(input_shapes[0], list)
            else None
        )
        groups = _reduction_groups(shape, semantic)
        combines = max(0, input_n - groups)
        add(
            "reduction",
            mode,
            combines,
            "input elements minus reduction groups",
        )
        if target == "mean":
            add("valu", mode, max(output_n, groups), "division per reduction result")
        elif target == "logsumexp":
            add(
                "sfu",
                mode,
                input_n + max(output_n, groups),
                "exponential and logarithm",
            )
        elif combines == 0:
            return {
                "model_version": RESOURCE_MODEL_VERSION,
                "work": {},
                "classification": "exempt",
                "exemption_reason": "degenerate_single_element_reduction",
                "formulas": [],
            }
    elif target in _SFU_OPS:
        add("sfu", mode, output_n, "one special-function result per output element")
    elif target in _COMPOSITE_SFU_OPS:
        add(
            "sfu",
            mode,
            output_n,
            "one nonlinear special-function result per output element",
        )
        add("valu", mode, 2 * output_n, "nonlinear scale/combine arithmetic")
    elif target in _INDEX_OPS:
        add(
            "valu",
            "integer",
            output_n,
            "one integer address/index operation per output element",
        )
    elif target in _VALU_OPS:
        add("valu", mode, output_n, "one vector ALU result per output element")
    elif target in {"pad", "constant_pad_nd", "roll", "tile", "flip"}:
        return {
            "model_version": RESOURCE_MODEL_VERSION,
            "work": {},
            "classification": "exempt",
            "exemption_reason": "memory_traffic_only",
            "formulas": [],
        }
    elif macs > 0:
        add(
            "mfma",
            _accumulation_mode(dtype, fallback_precision),
            2 * int(macs),
            "2 * contraction_macs",
        )
    elif strict:
        raise ResourceClassificationError(
            f"operation {target or '<missing>'!r} has no {RESOURCE_MODEL_VERSION} rule"
        )

    return {
        "model_version": RESOURCE_MODEL_VERSION,
        "work": {
            name: dict(sorted(modes.items())) for name, modes in sorted(work.items())
        },
        "classification": "modeled" if work else "unclassified",
        "exemption_reason": None,
        "formulas": formulas,
    }


def merge_resource_work(
    totals: dict[str, dict[str, int]], layer_work: Mapping[str, Mapping[str, Any]]
) -> None:
    """Add one layer's nested resource counters to graph totals."""
    for resource, modes in layer_work.items():
        target = totals.setdefault(str(resource), {})
        for mode, value in modes.items():
            target[str(mode)] = target.get(str(mode), 0) + int(value)


def validate_resource_work(value: Any) -> dict[str, dict[str, float]]:
    """Validate and normalize serialized resource counters."""
    if not isinstance(value, Mapping):
        raise ValueError("resource_work must be a mapping")
    normalized: dict[str, dict[str, float]] = {}
    for resource, modes in value.items():
        if not isinstance(modes, Mapping) or not modes:
            raise ValueError(f"resource_work.{resource} must be a non-empty mapping")
        normalized[str(resource)] = {}
        for mode, amount in modes.items():
            parsed = float(amount)
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError("resource work must be finite and non-negative")
            normalized[str(resource)][str(mode)] = parsed
    return normalized


__all__ = [
    "RESOURCE_MODEL_VERSION",
    "ResourceClassificationError",
    "classify_layer_resources",
    "merge_resource_work",
    "validate_resource_work",
]
