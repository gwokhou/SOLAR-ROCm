<!-- SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SOL-ExecBench SOLAR 能力缺口

本文记录 SOLAR-ROCm 相对 SOL-ExecBench 论文中 **SOLAR** 模块职责的
能力缺口和实施优先级。它不是 ROCm 评测沙箱、基线管理或反奖励黑客机制的
需求清单；这些属于 SOL-ExecBench 评测框架，而非 SOLAR 的三段式分析职责。

论文将 SOLAR 定义为：

1. 对给定输入形状执行 PyTorch 图提取；
2. 将运算图转换为经过数值验证的 extended-einsum 图；
3. 根据目标硬件的峰值吞吐、带宽、图级融合、预取和 Orojenesis 推导 SOL
   bound。

参见 [SOL-ExecBench, §4.2](https://arxiv.org/html/2603.19173v1#S4.SS2)。

## 当前结论

独立代码审查后，下面四项 P0 已在 `schema_version: 3` 的正式链路中关闭。
正式 workload 必须同时具备可执行语义图、source-to-einsum 数值 attestation、
合法融合证明、容量约束 Orojenesis 证据以及与 evaluator 相同的 AMD architecture
profile。任何一项缺少、陈旧、不适用或哈希/重算不一致，TSOL 和 SOL Score 都会
被拒绝。

这里的“P0 已关闭”指评分链对其声明的支持子集是 fail-closed 且证据可重放，
不表示已覆盖论文全部 workload。非矩阵资源模型、更多 AMD profile 和 official
corpus 覆盖率仍是下面的 P1；未覆盖的算子或无法安全组合的 multi-einsum 图不会
降级为近似正式分数。

## P0：先解决，否则 SOL 不可作为正式目标

### 1. 已关闭：容量约束 I/O lower bound、合法融合与预取

[`fusion.py`](../solar/analysis/fusion.py) 现在按依赖边构建保守融合区，并把
mutation、observable alias、atomic、多输出、同步归约、opaque library call 和
多 einsum 组合设为显式 barrier。每个 region 记录逐层级的峰值 live bytes、容量和
`capacity_pressure_bytes`；片上压力只作为诊断，不会被错误地直接计成 HBM spill。

[`orojenesis.py`](../solar/analysis/orojenesis.py) 运行固定 revision
`97d52178bf9a9c209bf79be96b87c164bcd35625` 的 Timeloop OAVES，保存 problem、
architecture、mapper、原始 curve 及 SHA-256。分析器按 last-level cache 容量选取
Pareto 点；仅当单 einsum 的 operand 可追溯至图外输入时，solver excess 才能与全图
compulsory I/O 安全组合。需要 multi-einsum solver 的内部 operand 不会被猜测；若
没有可组合层，正式分析直接失败。

正式字段语义为：`fused_bytes` 是去重的图外 compulsory I/O，
`io_lower_bound_bytes`/`prefetched_bytes` 再加入可审计的 tile-aware excess，
`lower_bound_seconds = max(compute_seconds, prefetched_memory_seconds)`。benchmark
loader 会重新生成 problem/config、重解析 raw curve、重算容量点、applicability、
I/O 与时间，并验证 evaluator profile 的 architecture hash 完全一致。

回归证据位于 `tests/test_p0_semantics_and_fusion.py` 和
`tests/test_source_to_sol_verification.py`，覆盖 barrier、容量压力、Pareto 选择、
curve 篡改、solver applicability、I/O/时间变化和 architecture 身份绑定。

### 2. 已关闭：official extended-einsum 语义可执行

[`semantics.py`](../solar/einsum/semantics.py) 定义 schema-v3 executable IR：保留有序
positional arguments、nested tensor arguments、kwargs、dtype/device/slice、显式输出
顺序，以及 mutation/alias/atomic/opaque effects。unknown-to-copy 回退已取消；未知
operation 要么作为参数完整的 exact ATen 调用执行，要么在 strict/official 模式失败。

[`einsum.py`](../solar/verification/einsum.py) 按该 IR 回放节点和整图，并同时比较
输出值与 dtype/shape、输入 mutation 和 storage alias 关系。受支持路径已覆盖带
dim 的 softmax、broadcast/视图、Attention mask/scale/causal、Norm、Embedding、
Gather/Scatter/索引、Conv、MoE 所需的 cat/stack/routing primitives、dtype conversion
和量化/反量化；多输出也按声明顺序执行。每个正式 workload 还必须通过三 seed ×
random/zero/boundary 的 hash-bound 整图 attestation，因此未覆盖 handler 无法进入评分。

对应节点/整图回归集中在 `tests/test_p0_semantics_and_fusion.py`，attestation 与官方
tolerance/nonfinite 策略回归位于 `tests/test_source_to_sol_verification.py` 和
`tests/test_rocm_timing_and_score.py`。

### 3. 已关闭：直接消费 benchmark reference 的动态 workload

[`sol_execbench.py`](../solar/benchmark/sol_execbench.py) 直接读取固定上游 schema
revision 的 `definition.json`/`workload.jsonl`，安全求值 axis expression，并按 UUID
为每个 workload 生成 random、scalar、custom 或 safetensors 输入。custom factory
只需返回 workload 显式标记的 custom 子集；未列出的输入按正式协议生成 random。
直接适配器与生成的 standalone `reference.py` 都校验 tensor shape/dtype/device，
并把无 shape 的随机输入保留为 Python scalar。

[`build_source_to_sol.py`](../solar/cli/build_source_to_sol.py) 对每个 UUID 独立执行 AMD
兼容性审计、ROCm eager 输入生成、`run()` 包装图提取、strict conversion、正式分析
和 attestation。任何阶段失败都会产生该 workload 的 stage-specific compatibility
证据，不会借用其他 workload 的图或静默 fallback。

`tests/test_sol_execbench_amd_compatibility.py` 覆盖多动态 shape、custom 子集、隐式
random、scalar、dtype drift、axis 严格性及 standalone round-trip。

### 4. 已关闭：精确 AOTAutograd joint backward graph

[`backward_processor.py`](../solar/graph/backward_processor.py) 的正式入口要求
PyTorch 2.11 AOTAutograd `trace_joint=True`，不再使用 best-effort FX/`grad_fn` 近似。
序列化结果保存 exact ATen nodes、真实 shape/dtype、有序 joint outputs、parameters、
buffers、user inputs/outputs、saved tensors、输入/参数梯度映射，以及 buffer/user-input
mutation 和 alias effects。

写盘后会重新加载 YAML，用同一 executable IR 回放 joint graph，并与 autograd
reference 比较所有 raw FX outputs、参数/输入梯度、argument mutation 和 storage alias
矩阵。任一不一致都会使提取失败。`tests/test_aot_joint_backward.py` 覆盖 linear、
softmax backward、norm-residual、MoE scatter/gather 以及 functionalized buffer
mutation。

## P1：覆盖扩大后会造成系统性偏差

### 5. 建立非矩阵操作的资源模型

当前性能模型统计 `other_ops`，但正式 compute cycle 只取
`compute_matrix_cycles` ([`perf_model.py`](../solar/perf/perf_model.py))。
因此 elementwise、softmax、归约、SFU、scan、sort、atomic 和整数索引的计算
不进入 compute-side bound。

需要按 AMD 资源类别建模 MFMA、VALU、SFU、reduction、atomic 和 scan/sort，
并计入低精度 scale、cast、dequantization 和 accumulation 成本。

### 6. 扩充 AMD hardware/precision profile，并加入峰值审计

仓库当前只随附 RX 9060 XT/gfx1200 profile
([`configs/arch/RX_9060_XT.yaml`](../configs/arch/RX_9060_XT.yaml))。
需要加入 MI300X、MI325X、MI350X 等目标，以及锁频下的各精度吞吐、HBM 带宽、
wave/MFMA 限制和 profile 来源。

拒绝 NVIDIA NVFP4 并不是缺陷：AMD profile 不应把不兼容格式冒充为 AMD FP4。
缺口是 AMD 原生 FP8 FNUZ、block scaling、accumulation dtype 和转换开销尚未
获得完整的语义与成本建模。详见 [`QUANT_SUPPORT.md`](QUANT_SUPPORT.md)。

### 7. 建立 AMD official corpus 与覆盖率门禁

当前 replayable source-to-SOL 示例集中于 Matmul；BERT/KernelBench 集成测试
主要验证能生成非空 graph/analysis，并不代表可签发整图数值 attestation。

应建立覆盖 BF16/FP8、forward/backward、Attention、Norm、MoE、SSM、Conv、
动态 shape 与结构化输入的 AMD corpus。每个 workload 应具备：

- 可回放的 source-to-SOL attestation；
- 独立推导的 FLOP/byte golden value；
- 正确实测内核低于 TSOL 时的自动审计；
- 按操作、dtype、前反向和动态路径统计的 fail-closed 覆盖率。

## P2：已知边界，不应抢占上述基础能力

### 8. 值依赖优化尚未建模

SOLAR 是 shape-based 分析，不能表达压缩、常量传播、重复数据、稀疏模式或
特殊 mask 带来的值依赖优化。这也是论文在 §4.2 明确说明的限制，应在 P0/P1
完成后再评估是否为特定 AMD workload 增加受控的 value-aware 模型。

## 建议实施顺序

P0 的实施顺序已经完成。后续按 P1 风险排序：

1. 建立非矩阵操作的 AMD 资源模型，避免 compute-side 系统性漏算；
2. 扩充并审计 MI300X/MI325X/MI350X 等 architecture/precision profile；
3. 建立 AMD official corpus 和按 operation/dtype/forward/backward/dynamic path 的
   fail-closed 覆盖率门禁。

因此项目可以声明“对 schema-v3 正式门禁所接受的 workload 提供可审计 SOLAR
bound”，但在 P1 corpus 和资源模型完成前，不应声明已覆盖论文的全部题目或全部
AMD 架构。
