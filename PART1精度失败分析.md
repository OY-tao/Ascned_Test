# PART1 精度测试失败分析

> 范围：PART1（test_shape 基线网格）精度（double-golden）校验结果
> 数据来源：8 台服务器 `/home/y00889330/{perf,acc}-part1-ep{8,16,32,64}/` 各 case 日志与 `*precision.json`
> 生成时间：2026-10-24

---

## 一、总体结论

| 档位 | Dispatch 精度 | Combine 精度 | 结论 |
|---|---|---|---|
| EP8（单机） | ✅ 通过（perf 日志 dg 正常） | ✅ 通过（bf16/fp8 各 2 case 均通过） | **通过** |
| EP16（双机） | ✅ 通过（perf 日志 dg 正常） | ✅ 通过（t4096/t16384 dg 均 passed） | **通过** |
| EP32（4 机） | ⚠️ 无精度数据（perf 无 dg；acc 未跑） | ❌ **全 rank 失败（double-golden）** | **失败** |
| EP64（8 机） | ⚠️ 无精度数据（perf 无 dg；acc 未跑） | ❌ **失败（t4096 SHMEM 组网 / t16384 double-golden）** | **失败** |

**核心问题：EP32/64 的 Combine 在 random 路由下出现「全 rank 数值错乱」，且这是结构性错误（非 FP8 舍入噪声）。EP8/16 同类校验全部通过，说明问题与多机（≥4 机）规模强相关。**

---

## 二、逐 case 失败明细

### 2.1 EP32 Combine（4 机 os1~4）

两个 tokens 档全部失败，失败点在 `route=random iter=1/2 combine` 的 double-golden 校验：

**T4096**（`perf1_combine_t4096_h7168_k6_e384`，elements=29,360,128）

| rank | actual max_relative | actual mean_relative | actual rmse | npu_baseline max_relative | ratios (max/mean/rmse) |
|---|---|---|---|---|---|
| 0 | 2.325e8 | 1081.6 | 9.10 | 6.25e5 | 372 / 2076 / 479 |
| 1 | 2.2e8 | 1145.4 | 9.08 | 4.74e5 | 464 / 1907 / 478 |
| 3 | 2.65e8 | 1139.2 | 9.07 | 6.25e5 | 424 / 1770 / 477 |
| 4 | 2.4125e8 | 1104.4 | 9.10 | 3.125e5 | 772 / 1960 / 479 |
| 5 | 2.2e8 | 1103.3 | 9.07 | 6.25e5 | 352 / 2024 / 478 |
| 7 | 2.65e8 | 1139.2 | 9.07 | 6.25e5 | 424 / 1770 / 477 |

**T16384**（elements=117,440,512）：同样全 rank 失败，actual max_relative≈2.86e8，mean_relative≈1064~1117，rmse≈9.1~9.13。

### 2.2 EP64 Combine（8 机 os1~8）

- **T4096**：未进入精度校验，`RuntimeError: aclshmemx_init_attr_with_buffers failed (status=-3)`（`/home/y00889330/perf-part1-ep64/perf1_combine_t4096_h7168_k6_e384.os1.log`）→ SHMEM 组网失败（跑前未清理 `/dev/shm/psm_*` 残留，偶发）。
- **T16384**：组网成功但 double-golden 失败（elements=117,440,512）：

| rank | actual max_relative | actual mean_relative | actual rmse | npu_baseline max_relative | ratios (max/mean/rmse) |
|---|---|---|---|---|---|
| 0 | 3.175e8 | 1515.7 | 12.08 | 6.25e5 | 508 / 3186 / 695 |
| 2 | 2.825e8 | 1555.7 | 12.09 | 6.25e5 | 452 / 3137 / 696 |
| 3 | 3.125e8 | 1510.7 | 12.08 | 4.69e5 | 667 / 2941 / 696 |
| 5 | 3.125e8 | 1510.7 | 12.08 | 4.69e5 | 667 / 2941 / 696 |
| 6 | 2.875e8 | 1593.2 | 12.07 | 6.25e5 | 460 / 3111 / 694 |
| 7 | 2.95e8 | 1580.5 | 12.10 | 3.91e5 | 755 / 3047 / 695 |

**全部 rank 的 max_relative 都是 ~2.2e8~3.2e8（≈ 2.5 亿），这不是噪声，而是 combine 输出里有大量元素与 golden 完全不匹配——典型的结构性错位（顺序/分组错误）。**

---

## 三、对照：通过侧的参考值（证明测试链路本身是好的）

EP8/EP16 同一套 double-golden 校验全部通过，数值特征：

| 档位 | actual max_relative | actual mean_relative | actual rmse | npu_baseline max_relative |
|---|---|---|---|---|
| EP8 t4096 | 3.9e5~4.5e5 | 0.41~0.45 | 0.0174 | 6.25e5 |
| EP8 t16384 | 3.9e5~6.3e5 | 0.48~0.52 | 0.0174 | 6.25e5 |
| EP16 t4096 | 3.1e5~3.9e5 | 0.33~0.38 | 0.0162 | 6.25e5 |
| EP16 t16384 | 3.1e5~4.7e5 | 0.27~0.33 | 0.0161 | 6.25e5 |

> 说明：`npu_baseline` 是**同一精度（BF16 NPU 参考实现）**的误差基线，数值 ~1e5~9e5 属 BF16 正常水平。EP32/64 的 actual 比它大 3~4 个数量级，且 rmse 达 9~12（EP8/16 只有 0.016），**远超出 BF16 舍入的合理范围**。

---

## 四、失败模式分析

### 4.1 为什么不是 FP8 舍入问题？

- 失败发生在 **combine**（BF16，`--dispatch-dtype bf16`），非 FP8 dispatch；
- 若是舍入误差，mean_relative 应为 ~0.3~0.5（EP8/16 实测值），这里却是 1000~1600；
- rmse 9~12 vs 正常 0.016，差 500~700 倍，远超阈值（ratio<2）。

**结论：combine 输出与 golden 的差异是「错误元素占大比例」的结构性错误，即输出错位、错序、漏算或重复计算。**

### 4.2 与规模的强相关性

| EP | 机数 | 节点内 rank | combine 结果 |
|---|---|---|---|
| 8 | 1 | 8 | ✅ |
| 16 | 2 | 8×2 | ✅ |
| 32 | 4 | 8×4 | ❌ 全 rank |
| 64 | 8 | 8×8 | ❌ 全 rank |

- EP8/16 与 EP32/64 的差异点：**跨机 SHMEM server 返回顺序**（`server_contribution_order` 中 `peer // 8` 分组）在多机规模下的实际 kernel 行为与 golden 假设不一致；
- EP32/64 的 `check_handle`（`metadata_version == 7`）在 EP8/16 上通过、EP32/64 未报 metadata 错误——**说明 metadata 层面正确，但 combine 的数值顺序在 ≥4 机规模下错了**；
- 失败同时出现在 **所有 rank 的第 1 轮 iter**（`iter=1/2`），说明是确定性错误（非随机抖动），可稳定复现。

### 4.3 可能的根因（按嫌疑排序）

1. **Notify server 返回顺序（server mode）在 ≥4 机下不一致**：`server_contribution_order` 按 `peer // 8`（即服务器 ID）升序分组求和，若 kernel 在多机（跨 SHMEM）时采用不同归约顺序（如 rank-first-hit 或按 rank 返回），则每个 server 的 partial 顺序错位，最终 combine 结果整体错乱。**这与「全 rank 同错、错误规模大、确定性」完全吻合。**
2. **SHMEM 跨机 session/notify 元数据一致性**：EP64 t4096 的 `aclshmemx_init status=-3` 说明跨机 SHMEM 会话本身偶发不稳定，EP32/64 的组合路径（双 SHMEM buffer）可能存在跨机数据竞态/错位。
3. **kernels `double_golden_outputs` 的 `rank_dedup` 语义**：测试按 `rank_dedup=False` 走 server 分组；若 kernel 实际启用了 rank dedup（重复 token 去重）而 golden 未模拟，输出会整体错位。

### 4.4 排除项

- ✅ **非 timeout 问题**：EP32 combine 每个 case 约 68s 完成（非卡死），EP64 t16384 约 263s 完成，均在校验后失败退出；
- ✅ **非 metadata v6/v7 不匹配**：`check_handle` 明确校验 `metadata_version==7`，EP8/16 通过；EP32/64 若版本错会在更早处报 AssertionError，而实际报的是数值错误；
- ✅ **非 FP8 引入**：combine 用 BF16；
- ✅ **非个别 rank 故障**：全部 8 rank 同时失败。

---

## 五、精度数据缺口（Dispatch 侧）

- **EP32/64 dispatch**：perf 脚本用 `--dtype fp8 --route random --repeat 2`，**不含 double-golden 校验**（日志中无 `double_golden` 记录）；
- **acc1 精度脚本**（`--route all --repeat 2`，含全部 7 种路由）**EP16/32/64 均未执行**（`acc-part1-*` 仅存在 EP8 目录）；
- EP8 的 acc1 dispatch 2 case 通过（both/all 各 7 分钟），但 combine 段因 `--route all` 的 double-golden 7 路由全跑（每 case ~15 分钟）导致 900s timeout 超时被杀（退出码 124/137）。

> 结论：**Dispatch 的精度（EP16/32/64）目前没有结论性数据**，需补跑 acc1_dispatch（至少 EP32/64）；combine 精度在 EP8/16 有明确通过证据，EP32/64 需先修复再补。

---

## 六、修复建议与下一步

### 优先（阻塞精度结论）
1. **定位 server 返回顺序差异**：在 os1 上以 EP32 单 case 复现（已可稳定复现），对比 `double_golden_outputs` 的 `server_contribution_order` 分组与 kernel 实际 `forward_list/rank_rows/gather_rows`，确认 ≥4 机时是否走 rank 模式或不同 server 归约；
2. **验证 `rank_dedup` 假设**：跑 `test_double_golden_server_order_matches_notify`（world=32/64, mode=random）单元测试，确认 golden 顺序假设本身是否正确；
3. **EP64 t4096 SHMEM**：重跑前先 `rm -f /dev/shm/psm_*` 清残留，确认是偶发还是必现。

### 次要
4. 补跑 **EP32/64 dispatch acc1**（`--route all`），补齐 dispatch 精度结论；
5. acc1 combine 的 timeout 从 900s 上调（单 case 实测 ~15min，需 ≥3600s），或拆分路由逐组跑；
6. 若确认是 kernel 顺序问题而非测试假设问题，向 DeepEP 上游反馈 ≥4 机 combine 的 Notify 顺序。

---

## 附：涉及日志与文件

| 路径 | 内容 |
|---|---|
| `/home/y00889330/perf-part1-ep32/perf1_combine_t4096_h7168_k6_e384.os1.log` | EP32 t4096 全 rank dg 失败详情 |
| `/home/y00889330/perf-part1-ep32/perf1_combine_t16384_h7168_k6_e384.os1.log` | EP32 t16384 全 rank dg 失败 |
| `/home/y00889330/perf-part1-ep64/perf1_combine_t4096_h7168_k6_e384.os1.log` | EP64 t4096 SHMEM status=-3 |
| `/home/y00889330/perf-part1-ep64/perf1_combine_t16384_h7168_k6_e384.os1.log` | EP64 t16384 全 rank dg 失败 |
| `/home/y00889330/perf-part1-ep{8,16}/perf1_combine_*.os1.log` | EP8/16 dg 通过（对照） |
| `tests/elastic/test_combine.py`（本地 DeepEP 源码） | double-golden 公式与 server_contribution_order |
