# PART1 — test_shape 基线网格（HT 版）汇总

> 生成时间：2026-10-25（基于 10-24 晚 ~ 10-25 晨补跑结果）
> 数据来源：os1 `/home/y00889330/perf-part1-ep8|ep16|ep32|ep64|ep64-rerun/`
> 基线来源：`test_shape.txt`（官方带宽基线 GB/s 区间，第 173~189 行）
> 统计口径：test_shape kernel 耗时（max rank）；带宽 = T×H×K×dtype_bytes / (t_us×1000)，算法带宽口径

---

## 一、覆盖总览

| 档位 | dispatch t4096 | dispatch t16384 | combine t4096 | combine t16384 |
|---|---|---|---|---|
| EP8 | ✅ | ✅ | ✅ | ✅ |
| EP16 | ✅ | ✅ | ✅ | ✅ |
| EP32 | ✅ | ✅ | ✅ | ✅ |
| EP64 | ⚠️ 仅 dispatch_t4096 补回 | ❌ 缺 | ❌ 缺 | ❌ 缺 |

**完成度：EP8/16/32 全通过；EP64 4 档中仅补回 1 档（dispatch_t4096），3 档仍缺。**

---

## 二、精度结果（rank json checks/passed）

| 档位 | dispatch t4096 | dispatch t16384 | combine t4096 | combine t16384 |
|---|---|---|---|---|
| EP8 | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| EP16 | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| EP32 | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| EP64 | ✅ PASS（补跑） | ❌ 原档失败 | ❌ 原档失败 | ❌ 原档失败（log only） |

**EP8/16/32 精度全 PASS；EP64 补跑的 dispatch_t4096 精度 passed=True，无回归。**

---

## 三、性能结果（实测 t_us / 带宽 vs 官方基线区间）

**统一配置**：random 路由，H=7168，K=6，E=384；dispatch FP8（dtype_bytes=1）、combine BF16（dtype_bytes=2），不传 topk_weights。

| EP | T | 算子 | 实测 t_us（max rank） | 实测 GB/s | 官方基线 GB/s | 判定 |
|---|---|---|---|---|---|---|
| 8 | 4096 | dispatch | 1132.0 | 155.6 | 210–224 | ▼ 低于（~74%） |
| 8 | 4096 | combine | 1132.1 | 311.2 | 211–228 | ▲ 高于 |
| 8 | 16384 | dispatch | 1684.9 | 418.2 | 217–231 | ▲ 高于 |
| 8 | 16384 | combine | 4354.4 | 323.6 | 215–225 | ▲ 高于 |
| 16 | 4096 | dispatch | 982.3（稳定轮 mean 703.7） | 179.3（mean 250.3） | 253–273 | ▼ 低于 |
| 16 | 4096 | combine | 777.4 | 453.2 | 249–268 | ▲ 高于 |
| 16 | 16384 | dispatch | 1611.4 | 437.3 | 255–281 | ▲ 高于 |
| 16 | 16384 | combine | 3022.9 | 466.2 | 253–268 | ▲ 高于 |
| 32 | 4096 | dispatch | 1135.0（稳定轮 mean 981.6） | 155.2（mean 179.5） | 365–378 | ▼ 低于 |
| 32 | 4096 | combine | 889.9 | 395.9 | 314–337 | ▲ 高于 |
| 32 | 16384 | dispatch | 1603.0 | 439.6 | 337–380 | ▲ 高于 |
| 32 | 16384 | combine | 3353.7 | 420.2 | 311–336 | ▲ 高于 |
| 64 | 4096 | dispatch | **2090.5（补跑）** | **84.3** | 370–395 | ▼ 低于（补跑数据） |
| 64 | 4096 | combine | ❌ 缺 | — | 337–349 | ⚠️ 无数据 |
| 64 | 16384 | dispatch | ❌ 缺 | — | 395–408 | ⚠️ 无数据 |
| 64 | 16384 | combine | ❌ 缺 | — | 364–376 | ⚠️ 无数据 |

> **判定汇总**：16 case 中 12 个有数据（dispatch 8 + combine 4）。
> - **高于官方基线（▲，8 档）**：EP8 combine×2、EP8/16/32 T16384 dispatch、EP16/32 combine×2 —— 其中 combine 全线高于基线（311~466 GB/s vs 211~376）；
> - **低于官方基线（▼，4 档）**：EP8/16/32 T4096 dispatch（155~179 GB/s vs 210~378）、EP64 T4096 dispatch 补跑（84.3 GB/s vs 370~395）—— 多机小 token 场景 dispatch 带宽偏低；
> - **无数据（4 档）**：EP64 combine×2 + EP64 dispatch_t16384（SHMEM status=-3/-4 失败）。

---

## 四、结论与待办

### 4.1 结论
1. **EP8/16/32 三个档位精度 + 性能数据齐全**，test_shape 口径基线网格基本建立；combine 全线高于官方基线，dispatch 低 EP 档（T4096）低于基线；
2. **EP64 补跑仅回补 dispatch_t4096**（2090.5us / 84.3GB/s），仍显著低于官方基线（370-395GB/s），其余 3 档仍为 SHMEM 组网失败（status=-3/-4）；
3. dispatch 首轮 vs 稳定轮差异大（fresh 调用），表中取稳定轮数据。

### 4.2 剩余缺口（高优先级）

| 缺口 | 严重度 | 说明 |
|---|---|---|
| PART1 EP64 dispatch_t16384 | 高 | 原档失败，补跑仍无数据 |
| PART1 EP64 combine_t4096 | 高 | 原档 SHMEM 失败，补跑未出 |
| PART1 EP64 combine_t16384 | 高 | 原档 SHMEM 失败，补跑未出 |

> **EP64 档位 4 缺 3，是全链路当前最大数据缺口**，建议优先排查 aclshmem 组网（-3 超时 / -4 端口冲突）后重跑。
