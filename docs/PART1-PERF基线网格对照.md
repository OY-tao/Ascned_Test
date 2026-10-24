# PART1-PERF：test_shape 基线网格（实测 vs 官方基线对照）

> 依据：`C:/Users/OY/Desktop/AITest/test_shape.txt`（第二部分性能基线，第 173~189 行）
> 实测来源：8 台服务器 `/home/deepep/perf-part1-ep{8,16,32,64}/` 各 case 日志的 kernel profile 数据
> 生成时间：2026-10-24

---

## 一、基线网格定义（官方归档）

**统一配置**：random 路由，H=7168，K=6，E=384；dispatch 用 FP8（dtype_bytes=1），combine 用 BF16（dtype_bytes=2）；均不传 topk_weights。

**网格规模**：4 个 EP 档 × 2 档 tokens（4096/16384）× 2 个算子（Dispatch/Combine）= **16 个 case**

| 档位 | EP | 机器数 | Dispatch FP8 基线 (GB/s) | Combine BF16 基线 (GB/s) |
|---|---|---|---|---|
| p1-ep8-perf | 8 | 1（os1） | T4096: **210–224** / T16384: **217–231** | T4096: **211–228** / T16384: **215–225** |
| p1-ep16-perf | 16 | 2（os1+2） | T4096: **253–273** / T16384: **255–281** | T4096: **249–268** / T16384: **253–268** |
| p1-ep32-perf | 32 | 4（os1~4） | T4096: **365–378** / T16384: **337–380** | T4096: **314–337** / T16384: **311–336** |
| p1-ep64-perf | 64 | 8（os1~8） | T4096: **370–395** / T16384: **395–408** | T4096: **337–349** / T16384: **364–376** |

> 文档另有 EP128/EP256 档（基线 305~325 / 159~189 GB/s），PART1 测试未覆盖。

---

## 二、实测结果（本次跑出的数据）

**统计口径**（对齐文档第 153 行）：每轮取各 rank 耗时的最大值，再对采样轮次求平均；带宽按文档公式 `T×H×K×dtype_bytes / (t_us×1000)`。

| EP | T | 算子 | kernel | 实测 t_us | 实测 GB/s | 官方基线 | 判定 |
|---|---|---|---|---|---|---|---|
| 8 | 4096 | dispatch | dispatch_kernel | 1132.0 | 155.6 | 210–224 | ▼ 低于 |
| 8 | 4096 | combine | combine_kernel | 1132.1 | 311.2 | 211–228 | ▲ 高于 |
| 8 | 16384 | dispatch | dispatch_kernel | 1684.9 | 418.2 | 217–231 | ▲ 高于 |
| 8 | 16384 | combine | combine_kernel | 4354.4 | 323.6 | 215–225 | ▲ 高于 |
| 16 | 4096 | dispatch | dispatch_kernel | 1223.3 | 144.0 | 253–273 | ▼ 低于 |
| 16 | 4096 | combine | combine_kernel | 777.4 | 453.2 | 249–268 | ▲ 高于 |
| 16 | 16384 | dispatch | dispatch_kernel | 1620.9 | 434.7 | 255–281 | ▲ 高于 |
| 16 | 16384 | combine | combine_kernel | 3051.0 | 461.9 | 253–268 | ▲ 高于 |
| 32 | 4096 | dispatch | dispatch_kernel | 3400.2 | 51.8 | 365–378 | ▼ 低于 |
| 32 | 4096 | combine | — | — | — | 314–337 | ⚠️ 无数据（case 失败） |
| 32 | 16384 | dispatch | dispatch_kernel | 3210.8 | 219.5 | 337–380 | ▼ 低于 |
| 32 | 16384 | combine | — | — | — | 311–336 | ⚠️ 无数据（case 失败） |
| 64 | 4096 | dispatch | dispatch_kernel | 1524.9 | 115.5 | 370–395 | ▼ 低于 |
| 64 | 4096 | combine | — | — | — | 337–349 | ⚠️ 无数据（case 失败） |
| 64 | 16384 | dispatch | dispatch_kernel | 3497.0 | 201.5 | 395–408 | ▼ 低于 |
| 64 | 16384 | combine | — | — | — | 364–376 | ⚠️ 无数据（case 失败） |

---

## 三、关键发现

### 1. Combine 的 EP32/EP64 数据缺失（4 case）
- **EP32 combine ×2**：`double-golden precision failed`（AssertionError，route=random iter1/2，rank3/7）
- **EP64 combine T4096**：`aclshmemx_init_attr_with_buffers failed (status=-3)` —— SHMEM psm 残留未清理（跑前未 `rm -f /dev/shm/psm_*`）
- **EP64 combine T16384**：274s 后同样精度失败
- 这 4 个 case **没有产出 kernel profile 数据**，需要修复后补跑

### 2. 带宽数据「偏高」的 4 个 case（▲）—— 口径需注意
- EP8 combine ×2（311/324 GB/s）、EP16 combine ×2（453/462 GB/s）、EP8/16 T16384 dispatch（418/435 GB/s）
- 均高于官方基线上限。原因：**实测带宽公式分子用的是 T×H×K，但 EP8/16 的每 rank payload 较小**，而日志中 combine 是 `combine_ep8_mte_kernel`（EP8 专用 MTE 路径），统计的 kernel 范围与文档口径可能存在差异（文档还提到 `combine_prepare_kernel` 单独记录不计入、`combine_reuse_kernel` 窗口复用单列）

### 3. Dispatch 的 EP32/64 明显偏低（▼）
- EP32 T4096 仅 51.8 GB/s（官方 365–378）—— 异常偏低
- EP32 T16384 219.5 / EP64 T4096 115.5 / EP64 T16384 201.5，均未达到官方下限
- 需排查：多机场景下 dispatch 的 kernel profile 是否受通信拥塞/大页/SHMEM 配置影响

### 4. Dispatch 首轮 vs 稳定轮差异巨大（fresh 调用）
示例（EP32 T4096 os1）：首轮 8 个 rank 为 2870~3368 us，稳定轮（第 2 轮）为 390~1135 us。
文档明确「使用已有 handle 测量 cached dispatch kernel；不包含 fresh 调用的 Notify」→ **取稳定轮统计是正确的口径**。

---

## 四、结论

| 统计项 | 数量 |
|---|---|
| 16 case 中已有数据 | 12（dispatch 8 + combine 4） |
| 达到官方基线（✅） | 0（▲ 偏高 4 个，▼ 偏低 4 个，均不在区间内） |
| 低于官方基线（▼） | 4（EP8/16 T4096 dispatch、EP32/64 dispatch 共 8 个 ▼，见上表） |
| 无数据 / case 失败（⚠️） | 4（EP32/64 combine） |

**结论**：PART1 网格 16 case 中，4 个 combine case（EP32/64）因精度/SHMEM 问题失败无数据，需修复重跑；8 个 dispatch case 中 EP32/64 明显低于官方基线（其中 EP32 T4096 仅 51.8 GB/s，异常），EP8/16 dispatch T4096 也低于基线；EP8/16 的 combine 与 T16384 dispatch 反而高于基线（需核对 kernel 统计口径）。**基线网格尚未完整达标**。
