# PART1-4 补跑方案（脚本集 + 缺口矩阵 + 执行说明）

> 生成时间：2026-10-24 18:20（18:35 更新：重跑数据独立存档，不覆盖原数据）
> 部署位置：os1 `/home/y00889330/rerun/`（77 个脚本：76 节点脚本 + 1 总控）
> 本地镜像：`C:/Users/OY/WorkBuddy/A5/rerun/` + 生成器 `gen_rerun_scripts.py` / `gen_rerun_extra.py`

---

## 〇、数据存档策略（重要）

**重跑（劣化>5%、原目录已有有效数据的补跑）一律输出到独立 `-rerun` 目录，不覆盖原数据**，方便新旧比对：

| 脚本 | 输出目录（重跑） | 原数据目录（保留） |
|---|---|---|
| `rerun_p1_ep64_os{1..8}.sh` | `/home/y00889330/perf-part1-ep64-rerun` | `/home/y00889330/perf-part1-ep64` |
| `rerun_p2_ep64_disp_os{1..8}.sh`（缺口） | `/home/y00889330/perf-ep64-dispatch-rerun` | `/home/y00889330/perf-ep64-dispatch`（11 档保留） |
| `rerun_p2_ep32_disp_deg_os{1..4}.sh`（劣化） | `/home/y00889330/perf-ep32-dispatch-rerun` | `/home/y00889330/perf-ep32-dispatch`（22 档保留） |
| `rerun_p2_ep64_disp_deg_os{1..8}.sh`（劣化） | `/home/y00889330/perf-ep64-dispatch-rerun` | `/home/y00889330/perf-ep64-dispatch`（11 档保留） |

**原目录本身完全没跑过（纯新增）的补跑，直接写原目录**（不存在覆盖问题）：
`perf-ep32-combine`、`perf-ep64-combine`、`acc-part1-ep32/64`、`acc-ep32/64-dispatch/combine`、k3fp8 等。

> 备注：调度器 18:29 的自动 `_rerun.sh` 只补原目录**无 summary 的缺口 case**（EP32 perf019 k3_t48、perf023 k3_t192），写原目录是正确补缺口，不覆盖有效数据。

---

## 一、缺口矩阵（现状盘点结论）

### 1.1 性能数据缺口

| PART | 档位 | 机器规模 | 状态 | 缺口 |
|---|---|---|---|---|
| PART1 | EP8/16 dispatch+combine | 1/2 机 | ✅ 有数据 | — |
| PART1 | EP32 dispatch+combine | 4 机 | ✅ 17:01/17:25 combine 重跑已通过 | — |
| PART1 | **EP64 dispatch t4096/t16384** | 8 机 | ❌ 16:40 重跑又失败（`aclshmem status=-3`） | **缺** |
| PART1 | **EP64 combine t4096/t16384** | 8 机 | ❌ t4096 SHMEM / t16384 精度失败 | **缺** |
| PART2 | EP32 dispatch ds 12 档 + k3 前 7 档 | 4 机 | ⏳ 调度器正在跑（k3 t192，剩 t256） | k3 t48~t256 补齐中 |
| PART2 | **EP32 combine 24 case** | 4 机 | ❌ 未跑 | **缺** |
| PART2 | EP64 dispatch ds 11 档 | 8 机 | ✅ 有数据 | — |
| PART2 | **EP64 dispatch ds_t8 + k3 12 档** | 8 机 | ❌ t8 SHMEM / k3_t1 内存崩 | **缺** |
| PART2 | **EP64 combine 24 case** | 8 机 | ❌ 未跑 | **缺** |
| PART3 | EP64 25 case | 8 机 | ✅ 22 PASS | — |
| PART3 | **EP64 高吞吐 BS4096 + K3 BS8/16** | 8 机 | ❌ 3 FAIL（会合超时/偶发） | **缺** |
| PART3 | **EP32 25 case** | 4 机 | ❌ 未跑 | **缺** |
| PART4 | EP64 dispatch/combine 4 档 | 8 机 | ✅ 有数据（`/home/b00943526/results/`） | — |
| PART4 | **EP32/16/8 sweep** | 4/2/1 机 | ❌ 未跑 | **缺**（调度器 PART4 `run_perf_sweep_all.sh` 自动补） |
| PART4 | **精度泛化 700 例/算子（EP64→32→16→8）** | 8→1 机拆组 | ❌ 未跑 | **缺**（调度器 PART5 `run_acc_all_eps.sh` 自动补） |

### 1.2 劣化超 5% 判定（PART2 dispatch，LL 口径 vs 归档「DeepEP低时延」）

**EP32 ds 组**（基线 17.07~72.96 μs）：

| BS | 实测 μs | 基线 μs | 偏差 | 判定 |
|---|---|---|---|---|
| 1~32 | 17.6~19.1 | 17.07~18.94 | ≤3.3% | ✅ |
| 48 | 29.61 | 25.73 | **+15.1%** | ❌ 重跑 |
| 64 | 32.92 | 27.76 | **+18.6%** | ❌ 重跑 |
| 96 | 40.40 | 32.70 | **+23.6%** | ❌ 重跑 |
| 128 | 48.42 | 41.84 | **+15.7%** | ❌ 重跑 |
| 192 | 65.55 | 59.89 | **+9.4%** | ❌ 重跑 |
| 256 | 80.98 | 72.96 | **+11.0%** | ❌ 重跑 |

**EP64 ds 组**（基线 19.54~80.77 μs）：

| BS | 实测 μs | 基线 μs | 偏差 | 判定 |
|---|---|---|---|---|
| 1~32/96~256 | 19.9~82.4 | — | ≤2.4% | ✅ |
| 48 | 31.25 | 28.44 | **+9.9%** | ❌ 重跑 |
| 64 | 33.57 | 31.03 | **+8.2%** | ❌ 重跑 |

> 注意：EP32 t48~t256 整体偏高（+9~24%），趋势一致，可能与 4 机小批量多机固有开销相关，重跑确认是否为真实劣化。

### 1.3 精度缺口（性能有、精度无）

| PART | 档位 | 状态 |
|---|---|---|
| PART1 | EP32/64 dispatch acc1（`--route all`） | ❌ 未跑 |
| PART1 | EP64 combine acc1（bf16/fp8） | ❌ 未跑（EP32 combine 已重跑通过） |
| PART2 | EP32/64 acc dispatch+combine（LL `--skip-perf-test` bf16/fp8） | ❌ 全部未跑 |

---

## 二、补跑脚本集（os1 `/home/y00889330/rerun/`）

### 2.1 性能缺口补跑

| 脚本前缀 | 规模 | 内容 | 子集过滤 |
|---|---|---|---|
| `rerun_p1_ep64_os{1..8}.sh` | 8 机 | PART1 EP64 dispatch/combine × t4096/t16384（HT） | `RERUN_ONLY="dispatch:4096,combine:16384"` |
| `rerun_p2_ep64_disp_os{1..8}.sh` | 8 机 | PART2 EP64 dispatch 缺口（ds_t8 + k3 12 档） | `RERUN_ONLY="ds:8,k3:1,...,k3:256"` |
| `rerun_p2_ep32_combine_os{1..4}.sh` | 4 机 | PART2 EP32 combine 24 case（LL+hccl） | `RERUN_ONLY="ds:1,k3:256"` |
| `rerun_p2_ep64_combine_os{1..8}.sh` | 8 机 | PART2 EP64 combine 24 case（LL+hccl） | 同上 |

### 2.2 劣化超 5% 重跑

| 脚本前缀 | 规模 | 内容 | 子集过滤 |
|---|---|---|---|
| `rerun_p2_ep32_disp_deg_os{1..4}.sh` | 4 机 | EP32 ds t48/t64/t96/t128/t192/t256 | `RERUN_ONLY="ds:48,ds:64,ds:96,ds:128,ds:192,ds:256"` |
| `rerun_p2_ep64_disp_deg_os{1..8}.sh` | 8 机 | EP64 ds t48/t64 | `RERUN_ONLY="ds:48,ds:64"` |

### 2.3 精度补跑（性能有、精度无）

| 脚本前缀 | 规模 | 内容 |
|---|---|---|
| `rerun_acc1_ep32_os{1..4}.sh` | 4 机 | PART1 acc1 EP32（dispatch both/all + combine bf16/fp8） |
| `rerun_acc1_ep64_os{1..8}.sh` | 8 机 | PART1 acc1 EP64（同上） |
| `rerun_acc_ep32_dispatch_os{1..4}.sh` | 4 机 | PART2 acc EP32 dispatch（LL bf16/fp8） |
| `rerun_acc_ep64_dispatch_os{1..8}.sh` | 8 机 | PART2 acc EP64 dispatch（LL bf16/fp8） |
| `rerun_acc_ep32_combine_os{1..4}.sh` | 4 机 | PART2 acc EP32 combine（LL bf16/fp8） |
| `rerun_acc_ep64_combine_os{1..8}.sh` | 8 机 | PART2 acc EP64 combine（LL bf16/fp8） |

### 2.4 总控

**`rerun_all.sh`** — 自动执行全部补跑：
- 检测 orchestrator_4part 调度器在跑则等待退出（不抢占）
- 每档先 kill_blockers 清 8 台残留 → 下发节点脚本 → 轮询完成
- done 标记：`~/.test_all_state/rerun.done/`（断点续跑）
- 用法：`bash rerun_all.sh`（全量）| `bash rerun_all.sh --only perf`（性能）| `--only deg`（劣化）| `--only acc`（精度）

---

## 三、执行顺序建议

1. **等当前调度器跑完 p2-ep32-perf-dispatch**（正在补 perf019/perf023 两档缺口）——它会自动续跑 p2-ep32-combine → k3fp8 → p2-ep64-dispatch/combine → PART3 → **PART4 MoonEP** → 精度阶段（PART1/2 acc → **PART5 MoonEP 精度泛化**）
2. 调度器跑完 PART2 后，**优先用总控补**：`bash rerun_all.sh --only perf`（补 EP64 缺口、combine，缺口写入 `-rerun` 目录）
3. 劣化重跑：`bash rerun_all.sh --only deg`（写入 `-rerun` 目录，原数据保留）
4. 精度补跑：`bash rerun_all.sh --only acc`（原目录无数据，直接写入）
5. **每档完成后用 `extract_part2_ep{32,64}.py` 分别提取新旧目录**，对照基线复核：
   - 旧（基线现场）：`perf-ep32-dispatch` / `perf-ep64-dispatch`
   - 新（重跑结果）：`perf-ep32-dispatch-rerun` / `perf-ep64-dispatch-rerun`
   - 逐档对比偏差，确认劣化是否消除

> ⚠️ **优先级约定**：性能缺口/劣化重跑 > 精度补跑 > **PART4 精度泛化（700 例/算子，优先级最低，最后执行）**
> 调度器 main() 顺序已固定为：PERF（PART1→2→3→4）→ ACC（PART1→2）→ PART5 MoonEP 精度泛化（最后）。
> 若精度泛化与更高优先级任务抢机器，应先停泛化、让位给性能/精度缺口。

> ⚠️ PART3/4/5 走各自的一键脚本（已集成进调度器，也可手动）：
> - PART3 EP32：os3 `bash run_perf_1click.sh --only-ep 32 --force`
> - PART3 FAIL case：`bash run_cases_multi.sh` 重跑（跳过已 PASS）
> - **PART4 MoonEP 性能**：os1 `bash /home/b00943526/run_perf_sweep_all.sh`（EP64→32→16→8，dispatch+combine）
> - **PART5 MoonEP 精度**：os1 `bash /home/b00943526/run_acc_all_eps.sh`（每算子 700 用例，`MAXC=2` 快速验证）
> - PART4 EP32/16/8：`EPLIST="32 16 8" OP=both bash run_perf_sweep_all.sh`

---

## 四、补跑执行结果（2026-10-25 11:30 复核）

> 全部补跑已于 10-24 晚 ~ 10-25 晨执行完毕（done 标记 11 个：rr_acc1_ep16/32、rr_p1_ep64、rr_p2_ep32_comb/deg、rr_p2_ep64_comb/deg/disp、rr_p3_ep16/32_perf、rr_p4_sweep）。

### 4.1 完成情况总表

| 补跑项 | 结果 | 说明 |
|---|---|---|
| PART1 EP64 dispatch/combine t4096/t16384 | ⚠️ 仅 dispatch_t4096 成功（2090.5us） | 其余 3 档仍缺（SHMEM -3/-4） |
| PART2 EP32 combine 24 case | ✅ 全完成（ds 12 独立 run + LL 双操作） | combine 与基线偏差 ≤1.8% |
| PART2 EP64 dispatch 缺口（ds_t8 + k3 12 档） | ✅ 全补齐 | ds_t8=21.26us；k3 12 档 22.9~171.2us |
| PART2 EP32 劣化重跑（t48~t256） | ✅ 6 档重跑完成，**劣化确认稳定** | 偏差 +9.8~23.5%，新旧 <1.5% |
| PART2 EP64 劣化重跑（t48/t64） | ✅ 2 档重跑完成，劣化确认 | +8.2%/+8.9% |
| PART2 EP64 k3 劣化复测（t48~t192，8 机） | ✅ 5 档复测完成（18:07-18:27） | 复测 vs 补跑偏差 <0.7%，**劣化稳定真实**（独立目录 k3retest 不覆盖原数据） |
| PART2 EP64 combine 24 case | ✅ 全完成 | 与基线偏差 ≤1.7%（t16 为 -13.6% 正向） |
| PART2 acc（EP32 dispatch bf16/fp8） | ✅ 11 case 全 PASS | acc011 fp8 缺单次 run |
| PART3 EP32 25 case | ✅ 25/25 全 PASS | 高吞吐 4590.7us vs 基线 4591.18（-0.01%） |
| PART3 EP64 FAIL case（3/43/44） | ⚠️ 未重跑 | case 3 有 10-23 试跑 4774.8us；43/44 偶发 |
| PART4 MoonEP EP8/16/32 sweep | ✅ 全完成（32 档） | 由 rr_p4_sweep done 标记确认 |
| PART4 acc 泛化（EP8/64 disp+comb） | ✅ 4/4 PASS | — |

### 4.2 更新后的缺口矩阵

| 剩余缺口 | 严重度 |
|---|---|
| PART1 EP64 dispatch_t16384 / combine_t4096/t16384 | 高（3 档） |
| PART2 acc011 fp8（ds t192） | 低 |
| PART3 EP64 case 3 高吞吐（正式轮） | 低（有试跑基线） |
| PART3 EP64 case 43/44（K3 BS8/16） | 低（偶发） |
| PART4 combine EP32 σ10/20、EP64 σ20 | 低（3 档未跑，其余已全线达标） |

> 详细数据见 `5-PART全量精度性能汇总（补跑后）.md`

### 4.3 更新记录（2026-10-25 19:05）

- **PART2 EP64 k3 dispatch 劣化档 8 机复测确认**（用户执行，18:07-18:27）：
  - 目录 `/home/y00889330/perf-ep64-dispatch-k3retest/`（独立新目录，不覆盖原数据），5 档全部 completed（ws=64）
  - 复测 vs 补跑值偏差 <0.7%：t48=47.91(-0.7%)、t64=56.72(+0.3%)、t96=73.22(-0.4%)、t128=93.59(+0.3%)、t192=131.56(-0.4%)
  - 结论：**劣化稳定真实**（vs 基线 fp8 10.19 高 +4.9%~+18.9%，基线为早期版本参考有限）
- **PART4 combine 结论修正**：补跑后全线达标（EP64 σ0.2/1.0=393.8/396.7GB/s=归档 386 的 102~103%），原"66-74% 短板"结论过时

### 4.4 更新记录（2026-10-26 10:30，PART1 EP64 诊断结论修正）

- **PART1 EP64 三档（dispatch_t16384 / combine_t4096/t16384）10-26 凌晨修复版重试仍全失败**（00:28 启动，8 台全下发，01:09 结束）
  - 修复项均已生效：os6 改经 **os1 跳板**成功参与（`relay_p1_ep64.log`：`os6 started (via os1)`）；master_wait **等满 7/7**（stdout：`ready=7/7, 放行`）；每档独立 waiting/go 标记防串扰
  - **os6 真实参与铁证**：其本地三档日志齐全（`/home/y00889330/perf-part1-ep64-rerun/*.os6.log`，10-26 01:02~01:15），aclshmem 日志 8 rank 全在，报错与 os1 一致（`WaitFinished timeout after 300000ms` → `status=-3`）
  - **结论修正**：根因 = **HT 版（test_dispatch.py/test_combine.py）8 机 64 rank SHMEM 组网 AllGather 必卡死**（与节点缺席/同步逻辑无关）；对照 LL 版（test_ep_low_latency.py + `--group-backend hccl`）同 8 机 64 rank 稳定成功（k3retest 5 档全 PASS）
  - 详细证据链见 `PART1_EP64_SHMEM诊断报告.md`（v2，含全量日志路径速查表）
- **修复方向（按优先级）**：① 给 HT 版补 `--group-backend hccl` 透传（对齐 LL 版成功路径）→ ② 4 机 32 rank 小规模对照验证 → ③ 兜底用 LL 版按 test_shape 口径补数据
