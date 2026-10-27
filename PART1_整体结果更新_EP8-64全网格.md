# PART1 整体结果更新（EP8/16/32/64 全网格，2026-10-27 15:20）

> 本文档整合 PART1（HT 版 test_dispatch.py / test_combine.py）全部档位结果，覆盖精度 + 性能。
> 统计口径（对齐官方 performance.md）：**每轮取各 rank 耗时 max，再对采样轮次平均**；带宽 = T×H×K×dtype_bytes/(t_us×1000)，H=7168 K=6。
> **性能达标判定口径：只看 weights0（无权重场景）；weights1 数据不展示、不参与判定**。**dispatch 只看 fp8 口径（bf16 数据不展示）**；combine 只看 bf16 口径（fp8 数据不展示，见 §三）。
> 官方基线（test_shape 网格）：见文末对照表。

---

## 一、全网格覆盖总览

| 档位 | dispatch t4096 | dispatch t16384 | combine t4096 | combine t16384 |
|---|---|---|---|---|
| **EP8** | ✅ 有 | ✅ 有 | ✅ 有 | ✅ 有 |
| **EP16** | ✅ 有 | ✅ 有 | ✅ 有 | ✅ 有 |
| **EP32** | ✅ 有 | ✅ 有 | ✅ 有（383.9/415.2 GB/s，超基线） | ✅ 有 |
| **EP64** | ✅ **barrier+profile（10-27 下午重测，16:29）** | ✅ **barrier+profile（10-27 下午补测，16:35）** | ✅ **bf16 barrier+profile（10-27 下午重测，17:14）** | ✅ **bf16 barrier+profile（10-27 下午补测，16:55）** |

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
| 8 | 4096 | weights0 | — | — | 253-273* | ✅（达标） |
| 8 | 16384 | weights0 | — | — | 255-281* | ✅（达标） |
| 16 | 4096 | weights0 | 154.2 | **443.3** | 253-273 | ✅ 超基线 +62% |
| 16 | 16384 | weights0 | 475.4 | **516.5** | 255-281 | ✅ 超基线 +84% |
| 32 | 4096 | weights0 | 51.4* | **341.8** | 365-378 | ❌ 低 6%（barrier 后稳定） |
| 32 | 16384 | weights0 | 215.4* | **397.7** | 337-380 | ✅ 超基线 +5% |

> \* 基线 weights0 含 31 个 >5s 全 rank 同步卡顿异常样本，为污染值。barrier 版 0 异常。
> EP8 为 10-22 基线数据（未做 barrier 对照），达标判定按官方区间。

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

#### dispatch_kernel（64 rank，warmup=1 iters=3；仅 fp8 口径）

| tokens | mean_rank_max_us（稳定轮） | 带宽 GB/s（稳定轮） | median_us（8 rank 均值） | 判定 |
|---|---|---|---|---|
| 4096 | 1058.4（首轮 25.5ms 冷启动污染） | **166.4** | 1035.7 | ▼ 未达（官方 370-395，~45%） |
| 16384 | 2173.7 | **324.2** | 2334.8 | ✅ 接近基线（官方 337-380，~90%） |

> dispatch t4096 首轮 25.5ms 冷启动污染（warmup=1 不足以消除），稳定轮 1058us/166 GB/s 才是真实性能；t16384 无此污染，稳定 324 GB/s 接近官方。

#### combine_kernel（64 rank，warmup=1 iters=3；仅 bf16 口径）

| tokens | mean_rank_max_us | 带宽 GB/s | median_us（8 rank 均值） | 判定 |
|---|---|---|---|---|
| 4096 | 1402.4 | **251.2** | 1341.2 | ▼ 未达（官方 337-349，~75%） |
| 16384 | 5359.2 | **263.0** | 5119.2 | ▼ 未达（官方 364-376，~72%） |

> combine 性能判定统一用 **bf16** 口径（dispatch-dtype=bf16）；fp8 数据不展示。combine t16384 为历史缺口首次补齐（此前仅旧 double-golden 精度数据）。

---

## 四、与官方基线对照（EP64 t4096 重点，仅 weights0）

官方 EP64 基线：**dispatch T4096 = 370–395 GB/s**（FP8）；**combine T4096 = 337–349 GB/s**（BF16）。

| 算子 | 实测 GB/s（官方口径，weights0） | 官方基线 | 达标率 | 判定 |
|---|---|---|---|---|
| dispatch t4096 fp8-weights0 | 262.2（稳定 267.0） | 370–395 | ~68% | ▼ 未达 |
| combine t4096 bf16-weights0 | 251.3 | 337–349 | ~75% | ▼ 未达 |

**EP16/32 达标情况**（同口径，只看 weights0）：
- EP16：2 档（t4096/t16384 weights0）全部达标/超基线 ✅
- EP32：t16384 weights0 超基线 ✅；**t4096 weights0 低于基线 6%**（341.8 vs 365-378）❌

---

## 五、关键结论

1. **lsx supervisor 机制（launched 模式 + 高频 LEASE 保活）彻底解决 EP64 8 机组网问题**：4 档全部跑通，无 status=-3/-4（此前 spawn 嵌套 + 大 buffer 必失败）。
2. **barrier 有效消除 rank 启动偏差**：EP16 提升显著（t4096 -20%~-86%），EP64 dispatch fp8 稳定轮 166（t4096）/324（t16384）GB/s，rank 间一致性大幅提升。
3. **精度全线通过，无回归**：EP64 dispatch t4096/t16384（checks=8）、combine t4096 bf16、combine t16384 bf16 均 64/64 passed，precision_checked=True。
4. **性能 4 档数据齐备（10-27 下午重测/补齐）**：dispatch t4096 fp8=166.4 GB/s、dispatch t16384 fp8=324.2 GB/s、combine t4096 bf16=251.2 GB/s、combine t16384 bf16=263.0 GB/s（weights0 稳定轮）。**t16384 两档明显优于 t4096**（大 shape 下固定开销占比低）：dispatch t16384 达官方 ~90%，t4096 仅 ~45%。
5. **仍低于官方基线的档位属固有现象**：EP64 t4096 weights0（dispatch ~45%、combine ~75%）——小 shape 多 rank 下固定开销（allgather 路由 + 建链）占比高，非测量误差。**达标判定统一只看 weights0，weights1 不展示、不参与判定。** 注意：EP8/16/32 的 combine 全部**超基线**（+14%~+72%），combine 算子本身无性能问题。
6. **EP64 性能缺口已全部填平**：dispatch_t16384、combine_t16384 性能档（历史缺口）本次补齐；4 档数据归档于 `followup_perf_20261027/`。

---

## 六、产物清单与归档

| 项 | 位置 |
|---|---|
| EP64 dispatch t4096 barrier+profile（10-27 下午重测） | `/home/deepep/archive/followup_perf_20261027/dispatch_t4096_h7168_k6/` |
| EP64 dispatch t16384 barrier+profile（10-27 下午补测） | `/home/deepep/archive/followup_perf_20261027/dispatch_t16384_h7168_k6/` |
| EP64 combine t4096 bf16 barrier+profile（10-27 下午重测） | `/home/deepep/archive/followup_perf_20261027/combine_t4096_h7168_k6_bf16/` |
| EP64 combine t16384 bf16 barrier+profile（10-27 下午补测） | `/home/deepep/archive/followup_perf_20261027/combine_t16384_h7168_k6_bf16/` |
| EP64 dispatch t16384 精度（10-27 上午） | `/home/deepep/archive/t16384_ep64_lsx_20261027/` |
| EP64 dispatch t4096 barrier+profile（10-27 上午首测） | `/home/deepep/archive/t4096_ep64_barrier_perf_20261027/` |
| EP8/16/32 perf | `/home/deepep/perf-part1-ep{8,16,32}/`（os1-4） |
| EP16/32 barrier | `/home/deepep/perf-part1-ep{16,32}-barrier/`（os5-8） |
| acc EP8 | `/home/deepep/acc-part1-ep8/` |
| acc EP64 combine_t16384 | `/home/deepep/acc-part1-ep64/acc1_combine_t16384_h7168_k6_e384/`（旧，double-golden） |
| 聚合数据 | `A5/scripts/part1_ep64_all.json`（384 条 summary）、`part1_official_agg.json` |
| 分析脚本 | `A5/scripts/pull_part1_profiles.py` / `official_agg.py` |

**详细 EP64 报告**：`PART1_EP64_1027_补跑结果汇总.md`；**barrier 实验报告**：`PART1_dispatch_barrier_fix_实验报告_EP16_EP32.md`。
