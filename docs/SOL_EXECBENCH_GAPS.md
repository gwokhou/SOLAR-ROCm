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

当前实现已具备这三个阶段的框架，并且对受支持的稠密前向工作负载可生成
可追溯的分析结果。`schema_version: 2` 工作负载还要求可回放的
source-to-SOL 数值 attestation；缺少、陈旧或不能执行的证明会导致 TSOL 和
SOL Score 被拒绝，而非静默接受。

这使其成为可信的受支持子集实现，但尚不能作为论文规模的正式 SOLAR：论文
包含动态形状、结构化输入、低精度、融合子图和反向工作负载，而当前覆盖和
bound 紧致性均不足。

## P0：先解决，否则 SOL 不可作为正式目标

### 1. 实现容量约束的 I/O lower bound、合法融合与预取

当前分析器不运行 Orojenesis：`orojenesis_elements` 为 `None`
([`graph_analyzer.py`](../solar/analysis/graph_analyzer.py))。同时，
`fused` 和 `fused_prefetched` 都使用相同的去重边界 I/O，因此得到相同的
内存量和运行时间。该限制已在 [`SOL_GUIDE.md`](SOL_GUIDE.md) 中明确记录。

正式 ROCm 评分直接使用分析产物中的 `macs_by_precision` 和 `fused_bytes`
计算 TSOL ([`evaluator.py`](../solar/benchmark/evaluator.py))。因此只修改
`perf_model.py` 不足以修复正式评分。

影响：全图中间张量被假设为完全保留在片上，得到的下界可能有效但过松、过于
乐观，不能表达 Attention、SSM、MoE 和多算子 L2 工作负载的可达数据移动量。

验收条件：

- 基于依赖、别名、mutation、多输出、归约、atomic 和库调用划分合法融合区；
- 对寄存器、LDS/L1/L2 容量与 spill/重读建立显式 I/O 模型；
- 接入 Orojenesis 或等价的 tile-aware I/O lower-bound 求解器；
- 将经审计的 bound 接入 `RocmEvaluator`，并区分 fused 和 prefetched 语义。

### 2. 将 official extended-einsum 从“结构检查”提升为“语义可执行”

转换器对没有专用 handler 的、但具有输入 shape 的操作存在通用 copy 回退
([`solar/einsum/analyzer.py`](../solar/einsum/analyzer.py))。`--official`
会检查 supportable 标记、方程和 dtype 元数据，但不能仅凭这些字段证明
方程语义正确。

source-to-SOL 验证器是当前的最终语义门禁，但它只执行一个很小的子集：
extended equation 中的 `()+-`、多输出层和多数结构化操作都被拒绝；softmax
固定在最后一维 ([`solar/verification/einsum.py`](../solar/verification/einsum.py))。

验收条件：

- official 模式中取消 unknown-to-copy 回退；
- IR 显式保存操作参数和语义，包括 dim、broadcast、scale、mask、causal、dtype
  conversion、alias 与 mutation；
- 执行器或等价验证后端覆盖 Attention、Norm、Embedding、索引/Scatter/Gather、
  Conv、MoE routing 与量化操作；
- 每个支持的 handler 同时具备节点级和整图级的数值回归。

### 3. 直接消费 benchmark reference 的动态 workload，而不是 Model-only 输入

论文中的问题以顶层 `run()` 为 reference，结构化输入由 `get_inputs()` 生成，
每题通常有约 16 个动态 workload。当前 `PyTorchProcessor` 要求
`Model`/`ReferenceModel` 和 `get_inputs()`，图提取优先使用 meta、失败后使用
CPU ([`pytorch_processor.py`](../solar/graph/pytorch_processor.py))。

影响：GPU-only FP8/custom op、设备相关控制流、paged KV cache、稀疏 mask 和
MoE routing 可能无法提取，或提取到不同于 ROCm 实际执行的路径。

验收条件：提供 benchmark-schema adapter：对每一个 workload 参数调用
`get_inputs(parameters, device)`，包装 `run()`，在 ROCm eager 路径或等价的
FakeTensor 路径提取图，并自动生成 graph、analysis 和 attestation。

### 4. 用精确 joint graph 替换 best-effort backward extractor

论文公开集有反向 workload，包括 MoE scatter、softmax backward 和
norm-residual backward。当前实现中，FX 路径明确不能直接 trace backward；
`grad_fn` 路径使用近似 shape，并会把输入梯度连到“第一个 backward op”
([`backward_processor.py`](../solar/graph/backward_processor.py))。

验收条件：使用 AOTAutograd/`torch.export` 的 joint forward-backward graph，
保留 saved tensor、梯度累加、alias/mutation、dtype 和真实 shape；对输入梯度和
参数梯度都进行端到端数值验证。

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

1. 固化 official 支持子集：取消语义 copy 回退，并将验证器扩展至 BERT/
   Attention/Norm/索引等基础图；
2. 实现合法融合区与 Orojenesis I/O bound，并让评分器消费该 bound；
3. 实现 schema-native workload adapter 和 AOTAutograd backward graph；
4. 扩充 AMD profiles、精度成本模型和 official corpus。

在第 1 至 3 项完成前，项目应表述为“受支持工作负载的 SOL 分析工具”，而不应
宣称为 SOL-ExecBench 级完整 SOLAR 替代实现。
