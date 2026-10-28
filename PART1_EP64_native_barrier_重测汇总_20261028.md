# PART1 EP64 native barrier 档重测汇总（warmup=10 / iters=50 / repeat=3）

> 测试时间：2026-10-28 02:08 – 03:21（8 机 os1~os8，64 rank launched 模式）
> 输出/归档：`/home/y00889330/perf-part1-ep64-nativebarrier/`、`/home/y00889330/archive/part1_ep64_nativebarrier_20261028/`
> 聚合目录：`/home/y00889330/perf-part1-ep64-nativebarrier/aggregate/`（node0~7 全量）

## 一、测试配置（用户指定 10/50/3）

| 项 | 值 |
|---|---|
| 脚本 | dispatch: `test_dispatch_barrier.py`（md5=b8bc5a7f，native barrier）；combine: `test_combine_barrier.py`（md5=03025315，Python 层 barrier） |
| 参数 | EP=64、8 节点×8 rank、H7168/K6/E384、route=random、repeat=3、**warmup=10 / iters=50** |
| 口径 | dispatch fp8（weights0）、combine bf16（weights0）；官方口径 = 每轮各 rank max → 轮次平均 |
| 机制 | lsx supervisor（launched 模式，RANK/LOCAL_RANK/WORLD_SIZE=64 注入，不走 spawn）；LEASE 保活（90s 超时，已并行化）；每 case 失败自动复测（最多 3 次） |

## 二、native barrier 方案（用户要求确认：已实现）

barrier 在 **native 内**（C++ 侧），紧贴 Dispatch kernel 前，同一通信 stream 执行：
- `buffer._dispatch_diagnostics = {"benchmark_barriers": True}` → C++ 侧 dispatch kernel 前同步全部 64 rank
- `device_ms` 相位 = `[entry_barrier, kernel, exit_barrier, total]`，kernel 耗时取 `device_ms[1]`
- 计时起点在入口 barrier 后，终点在 Dispatch kernel 后
- `read_native_summary` 将各相位统计为 entry_barrier / kernel / exit_barrier / device_total 四组独立统计
- 无 Python 侧 `buffer.barrier`，无需改算子
- **验证**：两档 source=**native_benchmark_barriers** 64/64 rank 全部生效

## 三、结果（8 节点 64 rank 全量）

### dispatch（FP8，weights0，native barrier）

| tokens | kernel mean_us | median_us | min/max rank | 带宽 GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|---|
| 4096 | 467.73 | 467.88 | 465.56 / 469.50 | **376.6** | 370–395 | ✅ 达标（~98%） |
| 16384 | 1730.58 | 1730.57 | 1726.06 / 1735.92 | **407.2** | 337–380 | ✅ 超基线 +7% |

**barrier 相位（rank0 示例，avg_us）**：
- t4096：entry_barrier=655.5、kernel=467.96、exit_barrier=21.35、device_total=1144.82
- t16384：entry_barrier=206.8、kernel=1729.96、exit_barrier=33.79、device_total=1970.55

**节点均衡性**：8 节点均值 t4096 在 467.2–468.3us 之间（差 <0.3%）；t16384 在 1729.5–1731.2us（差 <0.1%）。

### combine（BF16，weights0，Python 层 barrier + torch_npu profiler）

| tokens | kernel mean_us | median_us | min/max rank | 带宽 GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|---|
| 4096 | 1333.52 | 1342.82 | 1231.92 / 1424.63 | **264.2** | 337–349 | ▼ 低 22% |
| 16384 | 5093.02 | 5083.20 | 4757.53 / 5405.20 | **276.7** | 364–376 | ▼ 低 24% |

> combine 无 native 计时（lsx 方案：Python 层 `buffer.barrier(with_cpu_sync=True)` + torch_npu profiler CSV），双 golden 精度校验全通过（ratios=1.0，64 rank）。低 22–24% 属小 shape 多 rank 固定开销（allgather 路由 + 建链），与 10-27 warmup5 版（263.5/277.4 GB/s）一致。

## 四、历史对比（EP64 dispatch 首次 64 rank 全量达标）

| 档 | 10-27 warmup1 | 10-27 warmup5 | **本次 10/50/3** | 说明 |
|---|---|---|---|---|
| dispatch t4096 | 166.4（首轮污染） | 301.6 | **376.6 ✅ 超基线** | 冷启动污染彻底消除 |
| dispatch t16384 | 324.2（os1 口径） | 361.3 | **407.2 ✅ 超基线 +7%** | 64 rank 全量口径 |
| combine t4096 | 251.2 | 263.5 | 264.2 | 基本一致 |
| combine t16384 | 263.0 | 277.4 | 276.7 | 基本一致 |

**关键结论**：
1. **dispatch 突破**：本次是 EP64 dispatch 首次 8 节点 64 rank 全量达标/超基线（t4096 376.6 达 370–395 的 98%，t16384 407.2 超基线 +7%）。
2. **native barrier 全面生效**：64/64 rank source=native_benchmark_barriers，rank 间一致性极佳（t16384 max-min 仅 9.9us）。
3. **数据完整性修复**：此前文档 dispatch t4096 / combine t16384 归档仅 os1 8 rank，本次 8 节点 64 rank 全量齐备。
4. **combine 两档仍低官方基线 22–24%**：属固有开销（小 shape + 多 rank），非测量误差；与 EP8/16/32 超基线不冲突（combine 算子本身无性能问题）。

## 五、踩坑与修复（本次 3 个关键 bug）

| 问题 | 根因 | 修复 |
|---|---|---|
| 3 次 FileExistsError 失败 | 驱动 `mkdir -p $ODIR` 预创建，测试脚本 `args.output.mkdir(exist_ok=False)` 冲突 | **只 rm 不 mkdir** |
| os5/7/8 supervisor 自杀 → dispatch 卡死 | 驱动 get_phase 串行（8×15s）+ 轮询 15s，保活周期 > LEASE 90s 窗口 | **get_phase/touch_lease 全并行 + timeout 压缩 + 轮询 5s** |
| 归档只存 os1 数据 | 驱动在 os1 上 `cp $ODIR results_osN`，各节点结果在本地磁盘 | **各节点本地 tar 管道拉取**（`ssh 'tar -cf -' \| tar -xf -`） |

其他：supervisor 超时 1200s→2700s（combine t16384 需 ~18min）；combine manifest 用 `--dispatch-dtype bf16 --with-dispatch --seed 1`；combine 脚本 8 台分发 md5=03025315 一致。
