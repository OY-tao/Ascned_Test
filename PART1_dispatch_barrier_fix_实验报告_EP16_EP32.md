# PART1 dispatch barrier-fix 实验报告（后 4 机 os5-8）

**日期**：2026-10-26 21:54~22:09（数据全量复核 23:30）
**背景**：PART1 dispatch 性能"不达标"，疑似各 rank 算子启动时间不一致，导致 profiler 测量窗口内 kernel 提交后等待 peer 数据，测出的 kernel 时长偏大。
**修复**：在 `test_dispatch.py` profile invoke 中、`buffer.dispatch` 前插入 `buffer.barrier(with_cpu_sync=True)`（ElasticBuffer 原生 SHMEM 全局同步 barrier：当前流 sync + 通信流 sync + `native.barrier()` 全 rank 对齐）。

> ⚠️ 注意：`buffer.barrier()` 无参调用会抛 `NotImplementedError: Only the host-synchronous sequential barrier is implemented`（实现要求 `with_cpu_sync=True`），修复时必须显式传参。第一次跑 2 case 全失败（NotImplementedError），修正后全部通过。

## 一、实验设置

| 项 | 值 |
|---|---|
| 测试文件 | `deepep/tests/elastic/test_dispatch_barrier.py`（原文件补丁副本，SHA `0bc2dc6358aa4beb`，8 台一致） |
| EP16 | os5(master)+os6，2 机 × 8 rank = 16 rank |
| EP32 | os5(master)+os6+os7+os8，4 机 × 8 rank = 32 rank |
| 口径 | HT、random 路由、fp8 dispatch、h7168/k6/e384、warmup5/iters10/repeat2 |
| 输出 | `/home/deepep/perf-part1-ep16-barrier/`、`/home/deepep/perf-part1-ep32-barrier/`（各节点 node0-3） |
| 对照 | `/home/deepep/perf-part1-ep16/`、`/home/deepep/perf-part1-ep32/`（10-22 基线，os1-4） |
| 正确性 | 4 case 全部 checks=4 passed=True，日志 0 error |

## 二、全量结果（16/32 rank 全量，文档口径：每轮各 rank max 再对轮次平均）

> 说明：初版报告只用 os5 node0 的 8 个 rank（部分数据），本版拉取 4 节点全部 rank 重新核算。

> ⚠️ **判定口径**：性能达标只看 **weights0（无权重场景）**；weights1（带权重）数据**不展示、不参与判定**。

### EP16（16 rank 全量，仅 weights0）

| tokens | scenario | baseline t_us | barrier t_us | baseline GB/s | barrier GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|---|---|
| 4096 | weights0 | 1142.5 | **397.4** | 154.2 | **443.3** | 253-273 | ✅ 超基线 +62% |
| 16384 | weights0 | 1482.4 | **1364.2** | 475.4 | **516.5** | 255-281 | ✅ 超基线 +84% |

### EP32（32 rank 全量，仅 weights0）

| tokens | scenario | baseline t_us | barrier t_us | baseline GB/s | barrier GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|---|---|
| 4096 | weights0 | 3424.0* | **515.4** | 51.4* | **341.8** | 365-378 | ❌ 低于基线 94%（差 6%） |
| 16384 | weights0 | 3270.7* | **1771.9** | 215.4* | **397.7** | 337-380 | ✅ 超基线 +5% |

> \* = baseline weights0 含 31 个 >5s 异常样本（全 32 rank 中 31 个各 1 次 ~23.6s 卡顿），该值为被污染值。barrier 版无任何异常样本。

## 三、关键观察

1. **rank 间一致性大幅提升**：barrier 后 EP32 t4096 全部 32 个 rank 高度一致（400~540us，node0-3 均值 437~489us 无系统性差异）。baseline 各 rank 方差极大（EP32 t4096 weights0：2870~3368us 均值含 23.6s 卡顿）。这是"各 rank 启动时间不一致"的直接证据——被 barrier 消除。
2. **baseline weights0 卡顿被消除**：EP32 baseline 的 weights0（t4096/t16384）各 31 个 >5s 异常样本（全 rank 同步卡 ~18.7~23.6s 一次，与 EP8 单机同类问题），barrier 版 0 个异常。卡顿污染了 baseline 文档口径（max-per-iter），造成 baseline 严重虚低。
3. **改善集中在低 tokens / 小负载场景**：t4096 改善最大（-20%~-86%），t16384 改善小（±7%）；tokens 小时 kernel 自身耗时占比低，启动偏差占比高，barrier 效果显著。
4. **weights0/weights1 趋同**：baseline 中 weights0 明显偏大（EP32 t4096：3151 vs 982），barrier 后两者趋近。
5. **EP32 t4096 仍低于官方基线（差 6-11%）**：即使 barrier 消除启动偏差后，EP32 t4096 稳定测值 341.8/324.1 GB/s 仍未达官方下限 365 GB/s。这与 PART1-PERF基线网格对照.md 的结论一致（EP32 dispatch 小 shape 固有开销），非测量误差。t4096 每 rank 仅 4096 tokens 的固定开销在 32 rank 下占比高。

## 四、结论

1. **加 `buffer.barrier(with_cpu_sync=True)` 有效**：消除 rank 启动时间不一致后，PART1 dispatch 测值显著下降、rank 间趋于一致，且消除 baseline weights0 的全 rank 同步卡顿异常。确认 baseline 中"性能不达标"很大部分由测量启动偏差导致，而非算子本身。
2. **达标判定（文档口径全量，只看 weights0）**：
   - EP16：2 档 weights0 全部超基线（t4096 +62%、t16384 +84%）✅
   - EP32：t16384 weights0 超基线 +5% ✅；**t4096 weights0 仍未达官方下限（341.8 vs 365-378）**，差 6% ❌
3. **EP32 t4096 剩余差距属真实小 shape 开销**：barrier 后 rank 一致、无异常，341.8 GB/s 为稳定真实值。官方基线 365-378 GB/s 在 32 rank × 4096 tokens 的小负载下难以达到，属于 dispatch 固定开销（allgather 路由 + 建链）占比高的固有现象。
4. **下一步建议**：EP64 档脚本已备（`os{N}_run_perf1_dispatch_ep64_barrier.sh`，N=1..8），口径一致，可跑验证同效应；EP32 t4096 差距若需消除，需从算子层面优化（如减少固定开销、批量调度），非测试方法问题。

## 五、产物清单

- 测试文件：`deepep/tests/elastic/test_dispatch_barrier.py`（8 台，SHA `0bc2dc6358aa4beb`）
- 脚本：`/home/deepep/os{N}_run_perf1_dispatch_ep{16,32,64}_barrier.sh`（N 对应节点）
- 数据：`/home/deepep/perf-part1-ep16-barrier/`、`/home/deepep/perf-part1-ep32-barrier/`（os5-8 各节点）
- 提取 CSV：`/tmp/ep{16,32}_{baseline,barrier}.csv`（初版，仅 os5 node0 8 rank）+ 全量 JSON：`/tmp/dispatch_data/full/`（4 节点全 rank）
- 分析脚本：`extract_dispatch.py` / `compare_barrier.py` / `/tmp/agg_dispatch.py` / `/tmp/barrier_verdict.py` / `/tmp/rank_breakdown.py`
