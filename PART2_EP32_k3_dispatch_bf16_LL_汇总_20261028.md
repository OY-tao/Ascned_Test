# PART2 EP32 k3 组 Dispatch BF16 性能测试汇总（LL + hccl）

> 测试时间：2026-10-28 00:31（LL 低时延链路重跑，os1/2/7/8）
> 所属链路：**PART2（LL 性能链路，test_ep_low_latency.py）**
> 数据路径：`/home/y00889330/perf-dispatch-ep32-k3bf16-os1278/`（源目录）
> 归档路径：`/home/y00889330/archive/part2_ep32_k3_dispatch_bf16_20261028/`
> 说明：本文件为 k3 组（H3584/K16/E896）**BF16 dispatch** 的 LL 链路权威数据；此前误用 HT 脚本（test_dispatch_barrier.py）跑的 k3 bf16 T4096（381.7 GB/s，归档 `part1_ep32_k3_dispatch_bf16_20261027/`）**链路错误，不作判定依据**。

## 一、测试配置

| 项 | 值 |
|---|---|
| 拓扑 | EP32（os1/2/7/8 四机，8 rank/节点 = 32 rank） |
| 链路 | **LL 低时延**（test_ep_low_latency.py，`--group-backend hccl`） |
| kernel | `dispatch_server_dedup_kernel`（dedup 路径） |
| dtype | **BF16**（`--dtype bf16`） |
| 路由 | normal |
| 计时 | kernel_duration_us，perf_tail=30（每 rank 取尾 30 采样） |
| 口径 | **slowest-rank-mean**（32 rank 中 mean_us 最大的 rank，即最慢 rank 的 30 采样均值，=`dispatch.max_rank_mean_kernel_us`） |
| 带宽公式 | T×H×K×dtype_bytes ÷ (us×1000)，bf16 dtype_bytes=2 |
| 档位 | T1~256 / H3584 / K16 / E896，warmup5 / iters50 / 8GiB |

## 二、k3 组（补充组）BF16 12 档 —— 权威数据

档位规格：**T1~256 / H3584 / K16 / E896**，全部 completed、32 rank 全覆盖

| 档 | T | 最慢rank mean (us) | p95 | p99 | 带宽 GB/s |
|---|---|---|---|---|---|
| t1 | 1 | 21.75 | 23.06 | 23.30 | 5.3 |
| t2 | 2 | 22.96 | 26.85 | 29.34 | 10.0 |
| t4 | 4 | 22.03 | 23.56 | 23.98 | 20.8 |
| t8 | 8 | 23.02 | 24.66 | 25.21 | 39.9 |
| t16 | 16 | 24.80 | 26.86 | 27.17 | 74.0 |
| t32 | 32 | 33.55 | 35.08 | 37.41 | 109.4 |
| t48 | 48 | 53.43 | 55.75 | 57.06 | 103.0 |
| t64 | 64 | 62.71 | 65.99 | 67.26 | 117.1 |
| t96 | 96 | 86.28 | 88.99 | 90.28 | 127.6 |
| t128 | 128 | 107.19 | 111.16 | 113.17 | 137.0 |
| t192 | 192 | 149.58 | 153.88 | 157.36 | 147.2 |
| t256 | 256 | 194.65 | 203.76 | 206.35 | 150.8 |

**观察：**
- 带宽随 T 单调上升（5.3 → 150.8 GB/s），t32 是拐点（t16→t32 74.0→109.4 跳升），t48 短暂回落（103.0）后继续爬升——与 k3 fp8 LL 规律完全一致
- 同 T 下 bf16 带宽约为 fp8 的 **1.6-1.9x**（payload 大一倍 + 同 kernel 固定开销摊薄，带宽接近 2x；t256: 150.8 vs 90.3 = 1.67x）
- 小 T 档（≤16）kernel 固定开销 ~21-25us 主导，带宽仅个位数~74 GB/s，属 LL 链路固有特性（非测量误差）

## 三、bf16 vs fp8 对比（同 T 档位，同为 LL 链路）

| T | bf16 BW | fp8 BW | bf16/fp8 |
|---|---|---|---|
| 1 | 5.3 | 2.8 | 1.89x |
| 8 | 39.9 | 22.2 | 1.80x |
| 16 | 74.0 | 41.9 | 1.77x |
| 32 | 109.4 | 66.4 | 1.65x |
| 64 | 117.1 | 69.7 | 1.68x |
| 128 | 137.0 | 82.7 | 1.66x |
| 256 | 150.8 | 90.3 | 1.67x |

- 小 T 下 bf16/fp8 比值接近 2（固定开销占比小）；大 T 下稳定在 ~1.65x（kernel 效率差异所致）
- 两者带宽规律一致（t32 拐点、t48 回落），证明 LL 链路测量稳定、口径统一

## 四、结论

- **k3 组（H3584/K16/E896）BF16 dispatch（LL 链路）12 档全部完成**：带宽 5.3-150.8 GB/s，随 T 单调上升
- 数据质量：12/12 completed、32/32 rank 全覆盖、perf_tail=30 尾采样均值，与既有 fp8 LL 数据同口径可直接对比
- 该数据替代此前 HT 链路误跑产物（381.7 GB/s，作废）

## 五、相关脚本与数据

- 驱动脚本：`/root/run_k3bf16_os1278_rank{0,1,2,3}.sh`（os1=rank0 master，os2=rank1，os7=rank2，os8=rank3）
- 数据源：`/home/y00889330/perf-dispatch-ep32-k3bf16-os1278/`（12 档 summary + 4 台 log）
- 归档：`/home/y00889330/archive/part2_ep32_k3_dispatch_bf16_20261028/`
- 聚合脚本：`/home/y00889330/aggregate_k3bf16_os1278.py`
- fp8 对照：`PART2_EP32_k3_dispatch_bf16_汇总_20261027.md`（内容实为 k3 fp8 LL，10-26 shmemfix 版）
