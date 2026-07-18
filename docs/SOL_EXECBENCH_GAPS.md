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

P1.5 的非矩阵资源模型以及 P1.6/P1.7 的 **RX 9060 XT/gfx1200 范围**也已关闭：
profile v2 区分硬件能力、ROCm 库支持和 PyTorch 生产成熟度，正式锁频证据覆盖当前
软件栈可验证的各精度；固定 official corpus 的 14 个兼容 workload 全部正式
attested，NVFP4 保留为明确不兼容，schema-v2 覆盖门禁通过。MI300X、MI325X、MI350X
等其他硬件因当前没有可用实机按项目决策延期，因此不能据此声明“P1.6 对全部 AMD
型号已关闭”。

这里的“已关闭”指评分链对上述声明支持子集是 fail-closed 且证据可重放，不表示
已覆盖论文全部 workload 或全部 AMD 架构。当前除 canonical 线性 MatMul chain 外，
还支持经证明的 zero-copy layout/view bridge、共享权重 batch flatten 和 MatMul fanout
tree；其他 contraction、物化/条件 alias 或无法安全组合的 DAG 仍不会降级为近似正式
分数。

## P0：先解决，否则 SOL 不可作为正式目标

### 1. 已关闭：容量约束 I/O lower bound、合法融合与预取

[`fusion.py`](../solar/analysis/fusion.py) 现在按依赖边构建保守融合区，并把
mutation、observable alias、atomic、多输出、同步归约、opaque library call 和未经
证明的多 einsum 组合设为显式 barrier。canonical chain 以及经 endpoint、axis-map、
tile-shape 和 zero-copy alias 证明的扩展 MatMul region 才可越过这些边。每个 region
记录逐层级的峰值 live bytes、容量和 `capacity_pressure_bytes`；片上压力只作为诊断，
不会被错误地直接计成 HBM spill。

[`orojenesis.py`](../solar/analysis/orojenesis.py) 运行固定 revision
`97d52178bf9a9c209bf79be96b87c164bcd35625` 的 Timeloop OAVES，保存 problem、
architecture、mapper、原始 curve 及 SHA-256。分析器按 last-level cache 容量选取
Pareto 点；仅当单 einsum 的 operand 可追溯至图外输入，或经过含明确低精度
dequantization 的可重算预处理链时，solver excess 才能与全图 compulsory I/O 安全
组合。对 effect-free、单 dtype、binary MatMul region，分析器运行 pinned Orojenesis
multi-einsum FFMT sweep。canonical chain 要求相邻 output/input tile 完全兼容；扩展
region 还会重放精确 axis map，允许非条件 zero-copy view/transpose/permute、共享权重
的 broadcast batch flatten，以及端点完整的 MatMul fanout tree。联合点只收取图外输入、
各权重和最终输出流量，并以所有 mapping buffer 之和作为保守容量需求。物化 reshape、
条件 alias、混合 dtype、非 MatMul contraction 或不完整端点不会被猜测；若没有可组合
证明，正式分析直接失败。

正式字段语义为：`fused_bytes` 是去重的图外 compulsory I/O，
`io_lower_bound_bytes`/`prefetched_bytes` 再加入可审计的 tile-aware excess，
`lower_bound_seconds = max(compute_seconds, prefetched_memory_seconds)`。benchmark
loader 会重新生成 problem/config；对 multi-einsum 还会重建每个 FFMT mapper、重解析
所有 mapping-level raw curve、重新应用 axis map/分支 schedule、匹配 tile shape 并重组
joint Pareto curve；随后重算容量点、applicability、I/O 与时间，并验证 evaluator
profile 的 architecture hash 与 mapper toolchain identity 完全一致。

回归证据位于 `tests/test_p0_semantics_and_fusion.py` 和
`tests/test_source_to_sol_verification.py`，覆盖 barrier、容量压力、Pareto 选择、
curve 篡改、multi-einsum raw mapping 重签篡改、solver applicability、I/O/时间变化和
architecture/toolchain 身份绑定。`tests/test_orojenesis_multi_integration.py` 会运行
pinned mapper 的 single、canonical chain 与扩展 region；普通本地测试在未配置外部
toolchain 时跳过，而 Docker/GitHub 的 required 模式缺失 mapper 或执行失败都会直接
失败。

交付不再维护两个并行运行镜像。`docker/Dockerfile` 用固定 digest 的 Ubuntu stage
静态编译 mapper、生成 source tree/archive/binary/compiler-wrapper provenance，并在
`orojenesis-test` stage 执行真实 mapper 合约；最终只交付一个 ROCm/PyTorch 镜像，
其中仅复制 mapper binary 和 provenance，不携带 NVIDIA runtime、builder 或源码树。
PR CI 构建真实 mapper test stage，定时/手动 CI 另行构建并检查最终合并镜像。

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

## P1：RX 9060 XT 范围已关闭，其他硬件延期

### 5. 已关闭：建立非矩阵操作的资源模型

[`resources.py`](../solar/analysis/resources.py) 将 exact graph 节点 fail-closed 地分类为
MFMA、VALU、SFU、reduction、atomic、scan/sort 和 conversion，并按真实 tensor dtype
记录 `resource_work`。低精度 scale、cast、dequantization、variance/std reduction 和
FP32 accumulation 都显式进入计数；一元素退化 reduction 只作为命名 exemption，不
伪造计算量。

架构 profile 为每类资源提供有来源的保守上界。正式 compute-side bound 对同一资源
串行求和、跨独立资源取最大值，再与容量约束内存时间取最大值。任一节点未分类、
mode 未定义或资源覆盖不完整都会拒绝正式分析。锁频校准要求每个资源类别至少有
探针，且不能用实测结果替换理论分母。

### 6. RX 9060 XT 已关闭；其他 hardware profile 延期

仓库随附 RX 9060 XT/gfx1200 profile v2
([`configs/arch/RX_9060_XT.yaml`](../configs/arch/RX_9060_XT.yaml))。其
`precision_support` 为每种发布精度记录硬件原生性、ROCm/PyTorch 成熟度、校准策略
和官方来源；直观矩阵见 [`QUANT_SUPPORT.md`](QUANT_SUPPORT.md)。关键边界是：

- FP32/FP16/INT8 位于 ROCm 7.2 Radeon PyTorch 生产 datatype 范围；
- BF16 为 gfx1200 rocWMMA 原生且当前 PyTorch 实机可执行，但未列入该生产 datatype
  表，因此记录为库级/实机验证，不扩大成生产支持声明；
- gfx1200 原生 FP8 是 OCP E4M3/E5M2 输入、FP32 accumulation/output。两种编码均由
  `torch._scaled_mm` 实机验证并锁频校准，但该接口为 private 且生产表未列 FP8，故
  记录为部分/实验性框架支持；
- FNUZ 属于 gfx94x，不再错误别名到 gfx1200；NVFP4/通用 FP4 均 fail-closed；
- RDNA4 ISA 有 INT4 IU4 WMMA，但 ROCm 7.2 rocWMMA 没有 INT4 type，PyTorch 2.11 也
  没有已验证 gfx1200 INT4 matrix API。因此仅保留有来源的理论峰值，并以原因完整的
  schema-v3 exemption 免除实测，不声称软件支持。

正式证据
[`RX_9060_XT_resource_audit.yaml`](../configs/arch/evidence/RX_9060_XT_resource_audit.yaml)
在 `STABLE_PEAK` 锁频下覆盖 FP32/FP16/BF16、两种 OCP FP8、INT8、所有非矩阵资源和
HBM；所有 measured/upper-bound ratio 均未越界。profile 同时绑定证据 SHA-256。

MI300X、MI325X、MI350X 等 profile 的峰值、格式、软件矩阵与锁频证据本次明确延期；
未来必须各自独立完成，不能复用 gfx1200 的别名、探针或实测比率。

### 7. 已关闭：建立 AMD official corpus 与覆盖率门禁

[`RX_9060_XT_SOL_EXECBENCH.yaml`](../configs/corpus/RX_9060_XT_SOL_EXECBENCH.yaml)
固定 `nvidia/SOL-ExecBench` revision、Parquet/row/workload 哈希、独立 FLOP/byte/
resource golden 和目标 profile 原始哈希。`solar-build-sol-execbench-corpus` 可从固定
数据集批量重建完整 source-to-SOL artifact 与 `build-index.yaml`，无需在仓库提交
约 138 MiB 的可再生中间产物。

checked audit
[`RX_9060_XT_SOL_EXECBENCH_audit.yaml`](../configs/corpus/evidence/RX_9060_XT_SOL_EXECBENCH_audit.yaml)
的结果为：

- 15/15 workload 有终态兼容性证据且 `fallbacks_used: []`；
- 14/14 RX 9060 XT 兼容项都有可回放 source-to-SOL attestation，并通过独立 golden；
- official OCP FP8 block-scale workload 正式进入 `fp8->fp32` MFMA 与 scale/cast 成本；
- NVFP4 是唯一明确不兼容项，不做 dtype/shape/device/backend 替换；
- operation（Attention/Norm/MoE/SSM/Conv/MatMul）、dtype（FP32/BF16/FP8/FP16）、
  forward/backward、static/两档 dynamic shape、random/random-scalar/custom input 不仅
  有最低计数，还检查关键交叉组合；
- external footprint 分别覆盖 fits-L2、L2-to-LLC、exceeds-LLC，并用 M=1/M=8828 的
  fixed-shape GEMM pair 检查尺度变化；任一 deficit 使 coverage gate 失败；
- 每个 artifact 的 canonical architecture hash 必须等于 manifest 目标 profile，旧版或
  其他硬件证据不能混入；
- evaluator 已保留“正确实测内核低于 TSOL 即拒绝发布”的 bound-violation 审计。

本地 [`RX_9060_XT_CONFORMANCE.yaml`](../configs/corpus/RX_9060_XT_CONFORMANCE.yaml)
另行绑定接受/拒绝回归，覆盖 legacy chain、layout/view、batch、fanout、条件 alias、
artifact replay 和 toolchain tamper。它的 source 明确为 `repository_local`，不会进入
official 分母或被描述成上游 workload。

这关闭的是代表性 AMD corpus 与本地合约门禁，不是“论文全部题目已覆盖”的声明。

## P2：已知边界，不应抢占上述基础能力

### 8. 值依赖优化尚未建模

SOLAR 是 shape-based 分析，不能表达压缩、常量传播、重复数据、稀疏模式或
特殊 mask 带来的值依赖优化。这也是论文在 §4.2 明确说明的限制，应在 P0/P1
完成后再评估是否为特定 AMD workload 增加受控的 value-aware 模型。

## 后续实施顺序

本轮 1-4 项（toolchain/镜像、扩展 multi-einsum、schema-v2 official corpus、独立
conformance corpus）已经完成。后续工作不应回退当前门禁：

1. 在不混入 synthetic case 的前提下继续扩大 fixed official representative corpus；
2. 仅为具备完整端点、tile-shape 和容量证明的新 contraction/DAG 增加正式 composition；
3. 评估 P2 的 value-aware 模型；
4. CDNA profile/精度矩阵/锁频证据保持延期，直到有对应实机后逐型号建立，绝不从
   gfx1200 外推。

因此项目可以声明“对 RX 9060 XT schema-v3 正式门禁所接受的 workload 提供可审计
SOLAR bound，且 P1 representative corpus gate 已通过”；仍不应声明已覆盖论文的
全部题目、全部精度软件路径或全部 AMD 架构。
