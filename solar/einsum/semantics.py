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
        "__and__",
        "__invert__",
        "abs",
        "add",
        "addmm",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "batch_norm",
        "bitwise_and",
        "bitwise_not",
        "cat",
        "chunk",
        "clamp",
        "clone",
        "contiguous",
        "copy",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "cos",
        "cumsum",
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
        "float",
        "flatten",
        "gather",
        "gelu",
        "group_norm",
        "getitem",
        "hardsigmoid",
        "hardswish",
        "half",
        "identity",
        "index_add",
        "index_copy",
        "index_put",
        "index_select",
        "int",
        "layer_norm",
        "linear",
        "log",
        "log_softmax",
        "logsumexp",
        "long",
        "maximum",
        "matmul",
        "masked_fill",
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
        "repeat_interleave",
        "reshape",
        "rsqrt",
        "scaled_dot_product_attention",
        "scatter",
        "scatter_add",
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
        "to",
        "type_as",
        "unsqueeze",
        "view",
        "where",
        "zeros_like",
    }
)

_MUTATING_TARGETS = frozenset({"copy", "setitem"})
_ATOMIC_TARGETS = frozenset(
    {"index_add", "index_copy", "index_put", "scatter", "scatter_add"}
)
_LIBRARY_TARGETS = frozenset(
    {
        "batch_norm",
        "bfloat16",
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

# These ATen operations may return storage that aliases one of their inputs.
# Treat conditional aliases (for example ``reshape`` and ``contiguous``) as
# aliases as well: a formal fusion proof must be valid for every legal input
# layout, not only for the layout observed by one trace.
_ALIASING_TARGETS = frozenset(
    {
        "contiguous",
        "detach",
        "expand",
        "flatten",
        "getitem",
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
        "__and__": "bitwise_and",
        "__radd__": "add",
        "__rmul__": "mul",
        "__rpow__": "pow",
        "__rsub__": "sub",
        "__rtruediv__": "div",
        "__invert__": "bitwise_not",
        "sdpa": "scaled_dot_product_attention",
        "concat": "cat",
        "convtranspose1d": "conv_transpose1d",
        "convtranspose2d": "conv_transpose2d",
        "convtranspose3d": "conv_transpose3d",
        "__get__": "transpose",
        "__getitem__": "getitem",
        "max": "amax",
        "min": "amin",
        "t": "transpose",
        "type": "to",
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
    module_args = layer.get("module_args") or {}
    recorded_arguments = module_args.get("call_arguments")
    recorded_kwargs = module_args.get("call_kwargs")
    arguments = (
        _plain_value(recorded_arguments)
        if isinstance(recorded_arguments, list)
        else [{"tensor": index} for index in range(len(names))]
    )
    if (
        layer.get("is_real_einsum") is True
        and layer.get("einsum_equation")
        and layer.get("force_aten_semantics") is not True
    ):
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
    raw_target = str(layer.get("type", "")).lower().rsplit(".", maxsplit=1)[-1]
    if raw_target in {"__radd__", "__rmul__", "__rpow__", "__rsub__", "__rtruediv__"}:
        if len(arguments) != 2:
            raise SemanticGraphError(f"{raw_target} requires exactly two arguments")
        arguments = [arguments[1], arguments[0]]
    kwargs: dict[str, Any] = (
        _plain_value(recorded_kwargs) if isinstance(recorded_kwargs, Mapping) else {}
    )
    for source in (module_args, layer.get("additional_info") or {}):
        if isinstance(source, Mapping):
            for key, value in source.items():
                if key not in {
                    "call_arguments",
                    "call_kwargs",
                    "function_name",
                    "hierarchical_name",
                    "raw_attributes",
                    "training",
                } and not isinstance(recorded_kwargs, Mapping):
                    kwargs[str(key)] = _plain_value(value)
    if "dims" in kwargs and "dim" not in kwargs:
        kwargs["dim"] = kwargs.pop("dims")
    if target in {"view", "reshape"} and len(arguments) == 1 and "shape" not in kwargs:
        output_shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
        if len(output_shapes) == 1:
            # A fixed traced output shape completely specifies view/reshape.
            kwargs["shape"] = _plain_value(output_shapes[0])
    if target in {"softmax", "log_softmax"} and "dim" not in kwargs:
        raise SemanticGraphError(f"{target} requires an explicit dim")

    raw_layer_type = str(layer.get("type", ""))
    mutating = (
        target in _MUTATING_TARGETS
        or (raw_layer_type.endswith("_") and not raw_layer_type.endswith("__"))
        or layer.get("mutates_inputs") is True
    )
    aliases = list(_plain_value(layer.get("aliases") or []))
    if target in _ALIASING_TARGETS and arguments and not aliases:
        aliases = [
            {
                "output": 0,
                "input": 0,
                "conditional": target in {"reshape", "contiguous"},
            }
        ]
    overload = str(layer.get("overload", "default"))
    if (
        overload == "default"
        and target in {"std", "std_mean", "var", "var_mean"}
        and "dim" in kwargs
    ):
        overload = "correction" if "correction" in kwargs else "dim"
    return {
        "kind": "aten",
        "target": target,
        "overload": overload,
        "arguments": arguments,
        "kwargs": kwargs,
        "effects": {
            "mutates": [0] if mutating and arguments else [],
            "aliases": aliases,
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
        if not isinstance(kwargs, Mapping):
            raise SemanticGraphError(f"layer {layer_id} has invalid keyword arguments")

        referenced_tensors: set[int] = set()

        def collect_tensor_references(value: Any) -> None:
            if isinstance(value, Mapping):
                if "tensor" in value:
                    referenced_tensors.add(int(value["tensor"]))
                else:
                    for item in value.values():
                        collect_tensor_references(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect_tensor_references(item)

        collect_tensor_references(arguments)
        collect_tensor_references(kwargs)
        input_arity = len(names.get("inputs") or [])
        if any(index < 0 or index >= input_arity for index in referenced_tensors):
            raise SemanticGraphError(
                f"layer {layer_id} references a tensor outside its input metadata"
            )
        if input_arity and referenced_tensors != set(range(input_arity)):
            raise SemanticGraphError(
                f"layer {layer_id} does not preserve every ordered tensor argument"
            )

        effects = semantic.get("effects")
        if not isinstance(effects, Mapping):
            raise SemanticGraphError(f"layer {layer_id} lacks explicit effects")
        mutations = effects.get("mutates")
        aliases = effects.get("aliases")
        if not isinstance(mutations, list) or not isinstance(aliases, list):
            raise SemanticGraphError(
                f"layer {layer_id} has invalid mutation/alias effects"
            )
        if any(int(index) < 0 or int(index) >= input_arity for index in mutations):
            raise SemanticGraphError(f"layer {layer_id} has invalid mutation target")
        output_arity = len(names.get("outputs") or [])
        for alias in aliases:
            if (
                not isinstance(alias, Mapping)
                or int(alias.get("input", -1)) not in range(input_arity)
                or int(alias.get("output", -1)) not in range(output_arity)
            ):
                raise SemanticGraphError(f"layer {layer_id} has invalid alias effect")
        # Each item is (keyword spelling, positional arity that also proves the
        # parameter was preserved). Tensor references and literal arguments
        # share the ordered ``arguments`` list, matching the ATen call.
        required_parameters = {
            "chunk": (("chunks", 2),),
            "expand": (("sizes", 2),),
            "gather": (("dim", 3),),
            "getitem": (("item", 2),),
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
