# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned, fail-closed semantics for executable SOLAR graphs."""

# Semantic target classifications deliberately mirror the executor dispatch.
# pylint: disable=duplicate-code,too-many-locals,import-outside-toplevel,too-many-branches

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EINSUM_GRAPH_SCHEMA_VERSION = 3


class SemanticGraphError(ValueError):
    """A graph does not contain complete, executable operation semantics."""


SUPPORTED_ATEN_TARGETS = frozenset(
    {
        "abs",
        "add",
        "addmm",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "batch_norm",
        "cat",
        "chunk",
        "clamp",
        "clone",
        "contiguous",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "cos",
        "dequantize",
        "detach",
        "div",
        "elu",
        "embedding",
        "embedding_bag",
        "exp",
        "expand",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_tensor_affine",
        "flatten",
        "gather",
        "gelu",
        "group_norm",
        "hardsigmoid",
        "hardswish",
        "identity",
        "index_select",
        "layer_norm",
        "linear",
        "log",
        "log_softmax",
        "logsumexp",
        "maximum",
        "matmul",
        "mean",
        "minimum",
        "mish",
        "mm",
        "bmm",
        "mul",
        "narrow",
        "neg",
        "ones_like",
        "permute",
        "pow",
        "prod",
        "quantize_per_channel",
        "quantize_per_tensor",
        "relu",
        "repeat",
        "reshape",
        "rsqrt",
        "scaled_dot_product_attention",
        "scatter",
        "select",
        "sigmoid",
        "silu",
        "sin",
        "slice",
        "softmax",
        "split",
        "sqrt",
        "square",
        "squeeze",
        "stack",
        "sub",
        "sum",
        "tanh",
        "transpose",
        "unsqueeze",
        "view",
        "where",
        "zeros_like",
    }
)

_MUTATING_TARGETS = frozenset({"scatter", "index_copy", "index_put", "copy", "setitem"})
_ATOMIC_TARGETS = frozenset({"scatter", "index_copy", "index_put"})
_LIBRARY_TARGETS = frozenset(
    {
        "batch_norm",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "embedding_bag",
        "group_norm",
        "layer_norm",
        "linear",
        "scaled_dot_product_attention",
    }
)


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in value.items()
            if str(key) != "raw_attributes"
        }
    return str(value)


def _canonical_target(layer: Mapping[str, Any]) -> str:
    target = str(layer.get("type", "")).lower().rsplit(".", maxsplit=1)[-1]
    aliases = {
        "attention": "scaled_dot_product_attention",
        "sdpa": "scaled_dot_product_attention",
        "concat": "cat",
        "convtranspose1d": "conv_transpose1d",
        "convtranspose2d": "conv_transpose2d",
        "convtranspose3d": "conv_transpose3d",
        "getitem": "slice",
        "__getitem__": "slice",
        "max": "amax",
        "min": "amin",
        "t": "transpose",
        "to": "identity",
        "type": "identity",
    }
    # Dunder names carry semantic trailing underscores, whereas a trailing
    # underscore on an ATen operator denotes mutation.  Resolve aliases before
    # removing the latter so ``__getitem__`` cannot become ``__getitem``.
    if target in aliases:
        return aliases[target]
    return aliases.get(target.rstrip("_"), target.rstrip("_"))


def build_semantic_operation(layer: Mapping[str, Any]) -> dict[str, Any]:
    """Build the executable operation record for one cost-model layer."""
    if str(layer.get("type", "")).lower() == "start":
        return {"kind": "input", "target": "input", "arguments": [], "kwargs": {}}

    names = (layer.get("tensor_names") or {}).get("inputs") or []
    arguments = [{"tensor": index} for index in range(len(names))]
    if layer.get("is_real_einsum") is True and layer.get("einsum_equation"):
        return {
            "kind": "einsum",
            "target": "einsum",
            "equation": str(layer.get("einsum_equation", "")),
            "arguments": arguments,
            "kwargs": {},
            "effects": {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        }

    target = _canonical_target(layer)
    kwargs: dict[str, Any] = {}
    for source in (layer.get("module_args") or {}, layer.get("additional_info") or {}):
        if isinstance(source, Mapping):
            for key, value in source.items():
                if key not in {"raw_attributes", "training"}:
                    kwargs[str(key)] = _plain_value(value)
    if "dims" in kwargs and "dim" not in kwargs:
        kwargs["dim"] = kwargs.pop("dims")
    if target in {"view", "reshape"} and "shape" not in kwargs:
        output_shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
        if len(output_shapes) == 1:
            # A fixed traced output shape completely specifies view/reshape.
            kwargs["shape"] = _plain_value(output_shapes[0])
    if target in {"softmax", "log_softmax"} and "dim" not in kwargs:
        raise SemanticGraphError(f"{target} requires an explicit dim")

    mutating = target in _MUTATING_TARGETS or str(layer.get("type", "")).endswith("_")
    return {
        "kind": "aten",
        "target": target,
        "overload": str(layer.get("overload", "default")),
        "arguments": arguments,
        "kwargs": kwargs,
        "effects": {
            "mutates": [0] if mutating and arguments else [],
            "aliases": list(_plain_value(layer.get("aliases") or [])),
            "atomic": target in _ATOMIC_TARGETS,
            "opaque_library_call": target in _LIBRARY_TARGETS,
        },
    }


def annotate_semantics(graph: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    """Attach the latest semantic schema and optionally validate it strictly."""
    graph["schema_version"] = EINSUM_GRAPH_SCHEMA_VERSION
    for layer_id, layer in (graph.get("layers") or {}).items():
        if not isinstance(layer, dict):
            raise SemanticGraphError(f"layer {layer_id} is not a mapping")
        try:
            layer["semantic_op"] = build_semantic_operation(layer)
        except SemanticGraphError:
            if strict:
                raise
            layer["semantic_op"] = {
                "kind": "unsupported",
                "target": _canonical_target(layer),
                "reason": "operation parameters are incomplete",
            }
    if strict:
        validate_semantic_graph(graph)
    return graph


def validate_semantic_graph(graph: Mapping[str, Any]) -> None:
    """Validate the latest graph contract without accepting legacy schemas."""
    if int(graph.get("schema_version", 0)) != EINSUM_GRAPH_SCHEMA_VERSION:
        raise SemanticGraphError(
            f"einsum graph must use latest schema_version={EINSUM_GRAPH_SCHEMA_VERSION}"
        )
    layers = graph.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise SemanticGraphError("einsum graph has no layers")
    for layer_id, layer in layers.items():
        if not isinstance(layer, Mapping):
            raise SemanticGraphError(f"layer {layer_id} is not a mapping")
        shapes = layer.get("tensor_shapes") or {}
        dtypes = layer.get("tensor_dtypes") or {}
        names = layer.get("tensor_names") or {}
        for side in ("inputs", "outputs"):
            arity = len(shapes.get(side) or [])
            if (
                len(dtypes.get(side) or []) != arity
                or len(names.get(side) or []) != arity
            ):
                raise SemanticGraphError(
                    f"layer {layer_id} lacks explicit {side} name/shape/dtype metadata"
                )
        semantic = layer.get("semantic_op")
        if not isinstance(semantic, Mapping):
            raise SemanticGraphError(f"layer {layer_id} has no semantic_op")
        kind = str(semantic.get("kind", ""))
        if kind == "input":
            continue
        if kind == "einsum":
            equation = str(semantic.get("equation", ""))
            if not equation or "->" not in equation:
                raise SemanticGraphError(
                    f"layer {layer_id} has no exact einsum equation"
                )
            continue
        if kind != "aten":
            raise SemanticGraphError(f"layer {layer_id} is not executable exactly")
        target = str(semantic.get("target", ""))
        if target not in SUPPORTED_ATEN_TARGETS:
            import torch

            if not target.isidentifier() or not hasattr(torch.ops.aten, target):
                raise SemanticGraphError(
                    f"layer {layer_id} uses unsupported exact operation {target!r}"
                )
        if not isinstance(semantic.get("arguments"), list):
            raise SemanticGraphError(f"layer {layer_id} lacks explicit arguments")
        arguments = semantic.get("arguments") or []
        kwargs = semantic.get("kwargs") or {}
        # Each item is (keyword spelling, positional arity that also proves the
        # parameter was preserved). Tensor references and literal arguments
        # share the ordered ``arguments`` list, matching the ATen call.
        required_parameters = {
            "chunk": (("chunks", 2),),
            "expand": (("sizes", 2),),
            "gather": (("dim", 3),),
            "index_copy": (("dim", 4),),
            "index_select": (("dim", 3),),
            "log_softmax": (("dim", 2),),
            "logsumexp": (("dim", 2),),
            "narrow": (("dim", 4), ("start", 4), ("length", 4)),
            "permute": (("dims", 2),),
            "repeat": (("repeats", 2),),
            "select": (("dim", 3), ("index", 3)),
            "softmax": (("dim", 2),),
            "split": (("split_size_or_sections", 2),),
            "unsqueeze": (("dim", 2),),
        }
        missing = [
            key
            for key, positional_arity in required_parameters.get(target, ())
            if key not in kwargs and len(arguments) < positional_arity
        ]
        if target == "slice" and not (
            "dim" in kwargs and any(key in kwargs for key in ("start", "end", "step"))
        ):
            missing.append("explicit slice bounds")
        if missing:
            raise SemanticGraphError(
                f"layer {layer_id} lacks exact {target} parameters: "
                + ", ".join(missing)
            )


__all__ = [
    "EINSUM_GRAPH_SCHEMA_VERSION",
    "SUPPORTED_ATEN_TARGETS",
    "SemanticGraphError",
    "annotate_semantics",
    "build_semantic_operation",
    "validate_semantic_graph",
]
