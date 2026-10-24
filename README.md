# Ascned_Test

昇腾 DeepEP (Ascend NPU) 性能/精度测试数据归档。

## 目录结构

| 目录 | 内容 |
|---|---|
| `ep8/` `ep16/` | 早期 EP8/EP16 性能测试 summary（b080 基线） |
| `summary_pull_os2/` | os2 汇总数据 |
| `summary_w0rerun/` | EP8 weights0 重跑数据 |
| `part1_perf/` | PART1 基线网格测试（EP8/16/32/64，dispatch/combine） |
| `part2_perf/` | PART2 算子表测试（EP32 dispatch，EP32/EP64 其余档位待跑） |
| `docs/` | 测试 case 清单、基线网格对照、性能/精度分析文档 |

## 数据说明

- 每个 case 目录下仅保留 `summary.json`（kernel 耗时统计：avg/median/min/max + 原始 samples），原始 ASCEND_PROFILER_OUTPUT 不归档
- 路径结构保留服务器原始相对路径（`<批次>/home/deepep/...`），便于与服务器对应
- `test_shape.txt` 为官方基线（random 路由 · H=7168 · K=6 · E=384）

## 环境

- 昇腾 NPU + CANN 9.2.0 + SHMEM 1.7.0，conda 环境 `deepep-copy`（Python 3.12 + torch 2.10.0 + torch_npu 2.10.0.post4）
- 跑前需 `rm -f /dev/shm/psm_*`，多机需几乎同时启动，EP 必须为 2 的幂
