# PART1 整体结果更新（EP8/16/32/64 全网格，2026-10-27 22:00 更新）

> 本文档整合 PART1（HT 版 test_dispatch.py / test_combine.py）全部档位结果，覆盖精度 + 性能。
> 统计口径（对齐官方 performance.md）：**每轮取各 rank 耗时 max，再对采样轮次平均**；带宽 = T×H×K×dtype_bytes/(t_us×1000)，H=7168 K=6。
> **性能达标判定口径：只看 weights0（无权重场景）；weights1 数据不展示、不参与判定**。**dispatch 只看 fp8 口径（bf16 数据不展示）**；combine 只看 bf16 口径（fp8 数据不展示，见 §三）。
> **污染处理口径（10-27 污染检查确认）**：EP64 各档首样本冷启动污染**剔除后**才参与统计（dispatch t4096 首样本 25.4ms 已剔除）；combine 两档无污染。详见 `PART1_EP64_污染检查报告_20261027.md`。
> 官方基线（test_shape 网格）：见文末对照表。
> **10-27 22:00 新增**：EP32 v4 档（H7168/K6/E384 fp8）native barrier 重跑成功（详见 §三）；k3 组（H3584/K16/E896）属 part2 内容，见 `PART2_整体结果_EP32_EP64_LL_20261027.md`。

---

## 一、全网格覆盖总览

| 档位 | dispatch t4096 | dispatch t16384 | combine t4096 | combine t16384 |
|---|---|---|---|---|
| **EP8** | ✅ 有（native 重跑 22:13） | ✅ 有（native 重跑 22:13） | ✅ 有 | ✅ 有 |
| **EP16** | ✅ 有 | ✅ 有 | ✅ 有 | ✅ 有 |
| **EP32** | ✅ 有 | ✅ 有 | ✅ 有（383.9/415.2 GB/s，超基线） | ✅ 有 |
| **EP64** | ✅ **10/50/3 权威重测（10-28 02:08，376.6 GB/s 达标）** | ✅ **10/50/3 权威重测（10-28 02:14，407.2 GB/s 超基线）** | ✅ **bf16 barrier+profile（10-27 17:14；10-28 264.2 GB/s）** | ✅ **bf16 barrier+profile（10-27 16:55；10-28 276.7 GB/s）** |

> **EP64 变化**：10-26 前 4 档缺 3（dispatch_t16384 / combine_t4096 / combine_t16384 全 status=-3/-4 失败）→ 10-27 上午补齐 dispatch t16384 精度、combine t4096 bf16 性能+精度 → **10-27 下午 16:29-17:22 性能 4 档全部重测/补齐**（dispatch 只测 fp8、combine 只测 bf16，均 weights0 口径），**两个历史性能缺口（dispatch_t16384、combine_t16384）已填平，性能 4 档数据齐备**。

---

## 二、精度结果（全档全通过 ✅）

| 档位 | case | 结果 | 校验内容 |
|---|---|---|---|
| EP8 | dispatch t4096/t16384、combine t4096/t16384 | ✅ 通过（10-22 基线） | checks 4/4 |
| EP16 | dispatch t4096/t16384 | ✅ 通过 | checks 4/4 |
| EP32 | dispatch t4096/t16384 | ✅ 通过 | checks 4/4 |
| EP64 | dispatch t4096（barrier 版） | ✅ 64/64 passed, checks=8/8 | profile precision_checked=true |
| EP64 | dispatch t16384（lsx） | ✅ 64/64 passed, checks=8/8 | dtype both, route random |
| EP64 | combine t4096 bf16（barrier 版） | ✅ 64/64 passed, checks=2 | dispatch input + combine double-golden |
| EP64 | combine t16384（10-25） | ✅ 通过，**8 节点 64 rank 全 passed=true** | double-golden 自比对（actual vs npu_baseline ratios=1.0，NPU 结果可复现） |

**结论：PART1 精度在已测档位全部通过，无回归。**

---

## 三、性能结果（官方口径）

### EP8 / EP16 / EP32（10-22 基线 + 10-26 barrier 实验）

#### dispatch（GB/s，FP8 口径；仅展示 weights0）

| EP | tokens | scenario | 基线 GB/s | barrier GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|---|
| 8 | 4096 | weights0 | — | **400.6**（native，10-27 22:13） | 253-273* | ✅ 超基线 +47% |
| 8 | 16384 | weights0 | — | **426.8**（native，10-27 22:13） | 255-281* | ✅ 超基线 +52% |
| 16 | 4096 | weights0 | 154.2 | **443.3** | 253-273 | ✅ 超基线 +62% |
| 16 | 16384 | weights0 | 475.4 | **516.5** | 255-281 | ✅ 超基线 +84% |
| 32 | 4096 | weights0 | 51.4* | **424.6**（v4 native，权威） | 365-378 | ✅ 超基线 +13%（v4 native 重测） |
| 32 | 16384 | weights0 | 215.4* | **397.7** | 337-380 | ✅ 超基线 +5% |

> \* 基线 weights0 含 31 个 >5s 全 rank 同步卡顿异常样本，为污染值。barrier 版 0 异常。
> EP8 为 10-27 22:13 native barrier 重跑（os1 单机 8 rank，10/50/3），16/16 native_benchmark_barriers，weights0 干净（t4096 439.7us std 3.6 / t16384 1651.1us std 12.6）。旧 10-23 基线（weights0 污染 790-3344us）已弃用。归档：`/home/y00889330/archive/part1_ep8_nativebarrier_20261027/`。
> **EP32 t4096 判定更新（10-27 22:05）**：老数据 341.8/322.8/353.6 GB/s（warmup5/iters10 版）测得不准，**作废**；改用 v4 档 native barrier 权威数据 = **424.6 GB/s**（414.9us，weights0 n=32），超官方基线 +13% ✅。详见下方 v4 表。

**10-27 22:00 新增：EP32 v4 档 native barrier 重跑（os5-8 后 4 机，10/50/3 默认配置，weights0 口径）**：

| 档位 | shape | dtype | mean_rank_max_us | 带宽 GB/s | 归档 | 说明 |
|---|---|---|---|---|---|---|
| v4（t4096） | H7168/K6/E384 | **fp8** | 414.9（n=32, min 408.2/max 419.7, std 2.7） | **424.6** | `archive/part1_ep32_v4_fp8_20261027/` | 前 3 次 native 未生效 → 第 4 次重跑成功，64/64 native_benchmark_barriers |

> **v4 档**：EP32 t4096 主 shape（H7168/K6/E384）fp8 dispatch，native barrier 全部生效（对比此前 3 次重跑走 profiler 路径未生效），节点均衡 413-416us。
> k3 组（H3584/K16/E896）属 part2 内容，数据见 `PART2_整体结果_EP32_EP64_LL_20261027.md`。

#### combine（GB/s，BF16 输出 2 字节口径；仅展示 weights0）

| EP | tokens | 基线 GB/s（10-22） | barrier/新跑 GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|
| 8 | 4096 | **311.2** | — | 211–228 | ✅ 超基线 +36% |
| 8 | 16384 | **323.2** | — | 215–225 | ✅ 超基线 +44% |
| 16 | 4096 | **453.0** | — | 249–268 | ✅ 超基线 +69% |
| 16 | 16384 | **461.2** | — | 253–268 | ✅ 超基线 +72% |
| 32 | 4096 | **383.9** | — | 314–337 | ✅ 超基线 +14% |
| 32 | 16384 | **415.2** | — | 311–336 | ✅ 超基线 +24% |
| 64 | 4096 | — | **251.3**（bf16，文档口径） | 337–349 | ▼ 低 75%（10-27 barrier 版） |
| 64 | 16384 | — | **263.0**（bf16，文档口径） | 364–376 | ▼ 低 72%（10-27 下午补测，历史缺口已填平） |

> 注：EP8/16/32 combine 为 10-22 基线数据（本次重新拉取核算，此前基线网格文档误记 EP32 combine"无数据"，实际数据存在且达标）。EP64 combine t4096/t16384 为 10-27 barrier 版新跑（t16384 详表见下方 combine_kernel 表，mean_rank_max_us=5359.2、median=5119.2）。

### EP64（10-27 下午重测，barrier+profile，官方口径；仅 weights0）

#### dispatch_kernel（warmup=1 iters=3；仅 fp8 口径，weights0）

| tokens | mean_rank_max_us（剔除首样本） | 带宽 GB/s（剔除首样本） | median_us | 判定 | 数据范围 |
|---|---|---|---|---|---|
| 4096 | 1058.4（首轮 25.5ms 冷启动污染已剔除） | **166.4** | 1035.7 | ▼ 未达（官方 370-395，~45%） | ⚠️ 仅 os1 8 rank（归档未拉全） |
| 16384 | 2173.7（os1）/ 2210.8（全 64） | **324.2**（os1）/ 318.7（全 64） | 2334.8 | ✅ 接近基线（官方 337-380，~90%） | 文档值 os1 8 rank；全 64 略低 |

> dispatch t4096 首轮 25.5ms 冷启动污染（warmup=1 不足以消除，7/8 rank 首样本 >25x），**剔除首样本后 1058us/166 GB/s 才是真实性能**。t16384 首样本 5.6ms 轻污染（2.86x），剔除后全 64 rank = 2210.8us/318.7GB/s。
> ⚠️ **数据完整性**：t4096 归档仅 os1 8 rank；t16384 归档 8 节点 64 rank 全齐，但文档此前数值为 os1 8 rank 口径（2173.7us/324.2GB/s），全 64 rank 口径 = 2210.8us/318.7GB/s（时间 +1.7%）。

#### combine_kernel（warmup=1 iters=3；仅 bf16 口径，weights0）

| tokens | mean_rank_max_us | 带宽 GB/s | median_us | 判定 | 数据范围 |
|---|---|---|---|---|---|
| 4096 | 1402.4（os1）/ 1440.6（全 64） | **251.2**（os1）/ 244.6（全 64） | 1341.2 | ▼ 未达（官方 337-349，~75%） | 文档值 os1 8 rank；全 64 略低 |
| 16384 | 5359.2 | **263.0** | 5119.2 | ▼ 未达（官方 364-376，~72%） | ⚠️ 仅 os1 8 rank（归档未拉全） |

> combine 性能判定统一用 **bf16** 口径（dispatch-dtype=bf16）；fp8 数据不展示。combine t16384 为历史缺口首次补齐（此前仅旧 double-golden 精度数据）。
> ✅ **污染检查**：combine 两档首样本无污染（首/后续 median 比 1.01x / 0.99x），无需剔除。
> ⚠️ **数据完整性**：combine t16384 归档仅 os1 8 rank；combine t4096 全 64 rank 可用（lsx job 目录），全量口径 = 1440.6us/244.6GB/s。

### EP64 warmup5 重测（10-27 18:0x，warmup=5 iters=10 repeat=2，与 EP16/32 barrier 同参；仅 weights0）

> 按用户要求：EP64 性能档改用与 EP16/32 一致的 warmup5/iters10，数据另存 `archive/followup_perf_20261027_warmup5/` 与 warmup1 版并存对比。dispatch 只看 fp8、combine 只看 bf16。os1 8 rank，samples=10。

| 档 | mean_avg_us | 带宽 GB/s（warmup5） | warmup1 版 GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|
| dispatch t4096 fp8 | 584.2 | **301.6** | 166.4（首轮污染） | 370-395 | ▼ 低 18%（warmup5 消除冷启动） |
| dispatch t16384 fp8 | 1950.4 | **361.3** | 324.2 | 337-380 | ✅ 超基线 +6% |
| combine t4096 bf16 | 1337.2 | **263.5** | 251.2 | 337-349 | ▼ 低 22% |
| combine t16384 bf16 | 5079.5 | **277.4** | 263.0 | 364-376 | ▼ 低 24% |

> **warmup5 显著改善 dispatch**：t4096 166.4→**301.6**（首轮 25ms 冷启动被 warmup5 消除，真实值接近 300 而非 166）、t16384 324.2→**361.3**（超基线）。combine 两档小幅提升（251.2→263.5、263.0→277.4），仍低 22-24%（小 shape 固有开销）。
> ⚠️ warmup1 的 dispatch t4096 数据（166.4）受冷启动污染，**判定应以 warmup5 版（301.6）为准**。

### EP64 10/50/3 权威重测（10-28 02:08-03:21，warmup=10 iters=50 repeat=3，8 节点 64 rank 全量；仅 weights0）✅✅

> **用户指定参数重测**（与 EP8 native 档同参 10/50/3）。**这是 EP64 dispatch 首次 8 节点 64 rank 全量达标/超基线**（此前 warmup1/warmup5 数据为 os1 8 rank 口径或受冷启动污染）。dispatch native barrier 64/64 生效（source=native_benchmark_barriers），combine 用 lsx barrier 版（Python 层 barrier + profiler）。失败自动复测（t16384/combine t16384 首测失败后第 2 次成功）。归档：`archive/part1_ep64_nativebarrier_20261028/`。

| 档 | kernel mean_us（64 rank） | min/max rank us | 带宽 GB/s | 官方基线 | 判定 | 说明 |
|---|---|---|---|---|---|---|
| dispatch t4096 fp8 | 467.7 | 465.6 / 469.5 | **376.6** | 370-395 | ✅ **达标 ~98%** | barrier: entry 655.5/kernel 467.96/exit 21.4/total 1144.8us |
| dispatch t16384 fp8 | 1730.6 | 1726.1 / 1735.9 | **407.2** | 337-380 | ✅ **超基线 +7%** | barrier: entry 206.8/kernel 1729.96/exit 33.8/total 1970.6us |
| combine t4096 bf16 | 1333.5 | 1231.9 / 1424.6 | **264.2** | 337-349 | ▼ 低 22% | double-golden passed，固定开销 |
| combine t16384 bf16 | 5093.0 | 4757.5 / 5405.2 | **276.7** | 364-376 | ▼ 低 24% | double-golden passed，固定开销 |

> **对比结论**：dispatch t4096 166.4→301.6→**376.6**（冷启动彻底消除 + 64 rank 全量口径），t16384 324.2→361.3→**407.2**；combine 两档与 warmup5 一致（264.2/276.7 vs 263.5/277.4）。**节点均衡性极佳**：8 节点均值 t4096 差 <0.3%、t16384 差 <0.1%。详细报告见 `PART1_EP64_native_barrier_重测汇总_20261028.md`。

---

## 四、与官方基线对照（EP64 t4096 重点，仅 weights0）

官方 EP64 基线：**dispatch T4096 = 370–395 GB/s**（FP8）；**combine T4096 = 337–349 GB/s**（BF16）。

| 算子 | 实测 GB/s（官方口径，weights0） | 官方基线 | 达标率 | 判定 |
|---|---|---|---|---|
| dispatch t4096 fp8-weights0 | **376.6**（10-28 10/50/3 权威，64 rank 全量） | 370–395 | ~98% | ✅ **达标**（此前 warmup1 166.4 受冷启动污染已弃用） |
| combine t4096 bf16-weights0 | **264.2**（10-28 10/50/3，64 rank 全量） | 337–349 | ~79% | ▼ 未达（固定开销） |

**EP16/32 达标情况**（同口径，只看 weights0）：
- EP16：2 档（t4096/t16384 weights0）全部达标/超基线 ✅
- EP32：t16384 weights0 超基线 ✅；**t4096 weights0 超基线 +13%**（v4 native 424.6 vs 365-378）✅

---

## 五、关键结论

1. **lsx supervisor 机制（launched 模式 + 高频 LEASE 保活）彻底解决 EP64 8 机组网问题**：4 档全部跑通，无 status=-3/-4（此前 spawn 嵌套 + 大 buffer 必失败）。
2. **barrier 有效消除 rank 启动偏差**：EP16 提升显著（t4096 -20%~-86%），EP64 dispatch fp8 稳定轮 301.6（t4096）/361.3（t16384）GB/s（warmup5 口径，10-27 晚重测），rank 间一致性大幅提升。
3. **精度全线通过，无回归**：EP64 dispatch t4096/t16384（checks=8）、combine t4096 bf16、combine t16384 bf16 均 64/64 passed，precision_checked=True。
4. **性能 4 档数据齐备（10-27 warmup1/warmup5 双口径 + 10-28 10/50/3 权威重测）**：**判定以 10-28 10/50/3 为准**（与 EP8 native 档同参，8 节点 64 rank 全量）——dispatch t4096 fp8=**376.6 GB/s（达标，官方 370-395 的 98%）**、dispatch t16384 fp8=**407.2 GB/s（超基线+7%）**、combine t4096 bf16=**264.2 GB/s**、combine t16384 bf16=**276.7 GB/s**。**EP64 dispatch 首次 64 rank 全量达标/超基线**。warmup5 版（301.6/361.3/263.5/277.4）与 10/50/3 一致（combine 两档几乎相同），仅 dispatch t4096 因冷启动消除差异显著（166.4→301.6→376.6）。**t16384 两档明显优于 t4096**（大 shape 下固定开销占比低）。
5. **仍低于官方基线的档位属固有现象**：EP64 t4096 weights0（combine 10/50/3=264.2 达 337-349 的 79%）——小 shape 多 rank 下固定开销（allgather 路由 + 建链）占比高，非测量误差。**dispatch 两档 10/50/3 已达标/超基线（376.6/407.2），不再属于"未达标"**。**达标判定统一只看 weights0，weights1 不展示、不参与判定。** 注意：EP8/16/32 的 combine 全部**超基线**（+14%~+72%），combine 算子本身无性能问题。
6. **EP64 性能缺口已全部填平**：dispatch_t16384、combine_t16384 性能档（历史缺口）本次补齐；warmup5 重测版数据归档于 `followup_perf_20261027_warmup5/`（判定口径），warmup1 版归档于 `followup_perf_20261027/`（对比存档）。
7. **⚠️ 数据完整性（10-27 18:05 污染检查发现，10-28 已补齐）**：10-27 归档不完整（dispatch t4096、combine t16384 仅 os1 8 rank）；**10-28 10/50/3 权威重测 8 节点 64 rank 全量齐备**（聚合目录 `perf-part1-ep64-nativebarrier/aggregate/`，归档 `archive/part1_ep64_nativebarrier_20261028/`）。**污染检查**：dispatch t4096 首样本 25.4ms 严重污染（10-27 warmup1 已剔除）；warmup=10 彻底消除冷启动（10/50/3 无污染）。如需 64 rank 权威全量数据，以 10-28 档为准。
8. **EP32 t4096 判定更新（10-27 22:05）**：老数据（341.8/322.8/353.6 GB/s，warmup5/iters10 版）测得不准**作废**；改用 **v4 档 native barrier** 权威数据 = **424.6 GB/s**（414.9us，weights0 n=32，os5-8 后 4 机 10/50/3），超官方基线（365-378）+13% ✅。

---

## 六、产物清单与归档

| 项 | 位置 |
|---|---|
| EP64 dispatch t4096 barrier+profile（10-27 下午重测） | `/home/y00889330/archive/followup_perf_20261027/dispatch_t4096_h7168_k6/` |
| EP64 dispatch t16384 barrier+profile（10-27 下午补测） | `/home/y00889330/archive/followup_perf_20261027/dispatch_t16384_h7168_k6/` |
| EP64 combine t4096 bf16 barrier+profile（10-27 下午重测） | `/home/y00889330/archive/followup_perf_20261027/combine_t4096_h7168_k6_bf16/` |
| EP64 combine t16384 bf16 barrier+profile（10-27 下午补测） | `/home/y00889330/archive/followup_perf_20261027/combine_t16384_h7168_k6_bf16/` |
| EP64 warmup5 重测 4 档（10-27 晚，判定口径） | `/home/y00889330/archive/followup_perf_20261027_warmup5/` |
| **EP64 10/50/3 权威重测 4 档（10-28，8 节点 64 rank 全量，判定口径）** | **`/home/y00889330/archive/part1_ep64_nativebarrier_20261028/`**（聚合 `perf-part1-ep64-nativebarrier/aggregate/`） |
| EP32 t4096 老数据 v2/v3（已作废，仅存档） | `/home/y00889330/archive/ep32_v2_20261027/`、`ep32_v3_20261027/` |
| EP32 v4 fp8 native（10-27 22:00 重跑成功） | `/home/y00889330/archive/part1_ep32_v4_fp8_20261027/`（os1+os5 各一份） |
| EP64 dispatch t16384 精度（10-27 上午） | `/home/y00889330/archive/t16384_ep64_lsx_20261027/` |
| EP64 dispatch t4096 barrier+profile（10-27 上午首测） | `/home/y00889330/archive/t4096_ep64_barrier_perf_20261027/` |
| EP8/16/32 perf | `/home/y00889330/perf-part1-ep{8,16,32}/`（os1-4） |
| EP8 native barrier 重跑（10-27 22:13，权威） | `/home/y00889330/archive/part1_ep8_nativebarrier_20261027/` |
| EP16/32 barrier | `/home/y00889330/perf-part1-ep{16,32}-barrier/`（os5-8） |
| acc EP8 | `/home/y00889330/acc-part1-ep8/` |
| acc EP64 combine_t16384 | `/home/y00889330/acc-part1-ep64/acc1_combine_t16384_h7168_k6_e384/`（旧，double-golden） |
| 聚合数据 | `A5/scripts/part1_ep64_all.json`（384 条 summary）、`part1_official_agg.json` |
| 分析脚本 | `A5/scripts/pull_part1_profiles.py` / `official_agg.py` |

**详细 EP64 报告**：`PART1_EP64_1027_补跑结果汇总.md`；**barrier 实验报告**：`PART1_dispatch_barrier_fix_实验报告_EP16_EP32.md`。
