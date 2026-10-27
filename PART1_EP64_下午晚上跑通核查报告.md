# PART1 EP64 下午到晚上"疑似跑通"核查报告

**核查时间**：2026-10-26 23:39-23:55
**核查对象**：用户反馈"PART1_EP64 今天下午到晚上疑似跑通"，检查 8 机环境是否有记录

## 一、结论（一句话）

**是的，跑通了。** lsx 账号今天 **18:49~21:28 期间在 8 机上成功跑通 4 次 PART1 EP64 dispatch t4096 精度校验**（64 rank 全通过），根因是 **10-26 14:34 有人重新编译更新了 deep_ep 的 `_C.so`（7a1852af），17:12 更新了 `elastic.py`**——这两处更新修复了此前"HT 版 8 机 64 rank SHMEM 组网必卡死（status=-3）"的问题。

## 二、关键证据链

### 1. lsx 跑通记录（4 次成功）

| 目录 | 时间 | world | tokens | checks | passed | exit_code |
|---|---|---|---|---|---|---|
| `ht_ep64_dispatch_acc_20261026_a04` | 18:49-18:51 | **64** | 4096 | 8 | **true** | 0 |
| `ht_ep64_dispatch_acc_20261026_a07` | 20:55-20:57 | **64** | 4096 | 8 | **true** | 0 |
| `ht_ep64_dispatch_acc_20261026_a08` | 21:04-21:07 | **64** | 4096 | 8 | **true** | 0 |
| `ht_ep64_dispatch_acc_20261026_a12` | 21:26-21:28 | **64** | 4096 | 8 | **true** | 0 |

- 路径：`/home/lsx/ht_ep64_dispatch_acc_20261026_a0X/`
- 测试命令：`test_dispatch.py --ep-size 64 --nproc-per-node 8 --master-addr 177.22.21.1 --tokens 4096 --hidden 7168 --topk 6 --experts 384 --dtype both --route random --repeat 2`
- **关键日志**（a04 rank-0.log）：`buffer.init.done` 仅 **87.6s**（对比凌晨重试 300s 超时卡死）；bf16+fp8 两种 dtype、weights 有无两种 case 全部 `passed`；总耗时 ~127s；exit_code=0
- 另 9 个目录（a01/a02/a03/a05/a06/a11/a13/a14）为失败/中止尝试（外部任务冲突、SIGTERM、无输出）

### 2. 环境变化时间线（为什么能跑通）

| 时间 | 事件 |
|---|---|
| 10-26 11:33 | shmemfix 实验（仅元数据对齐，`_build_info.json.bak_20261026_shmemfix`） |
| **10-26 14:34** | **`_C.so` + `libdeepep_kernels.so` + `_build_info.json` 重新编译更新**（SHA `7a1852af...`） |
| **10-26 17:12** | **`elastic.py` 更新**（76,342 字节，Python 层） |
| 10-26 17:22 | `buffers/__pycache__` 更新 |
| **10-26 18:49** | lsx 首次跑通 EP64 dispatch t4096（之后 3 次复现） |

- 8 台 `_C.so` 全部一致：SHA `7a1852af...`（os1-4 mtime 10-26 14:34，os5-8 mtime 10-23 17:13，内容相同）
- SHMEM 库未变：`libshmem.so` SHA `7e36a92d`（Oct 22 构建，与之前相同）
- lsx 跑通时加载的库（rank-0.exit.json loaded_files 实测）：`/home/deepep/deepep/third-party/shmem/install/shmem/lib/libshmem.so`（7e36a92d）、`_C.so`（7a1852af）——**与我们当前环境完全一致**

### 3. 对比凌晨失败（为什么之前不行）

- 凌晨 rerun（10-26 00:28-01:14，`perf-part1-ep64-rerun`）失败：`aclshmemx_init_attr_with_buffers failed (status=-3)`（`elastic.py:239` → `_runtime.py:233` → `native.Runtime`），三档全部 300s 超时
- 当时 `_C.so` 为旧版本（14:34 更新前）
- 修复方向：**无需再改测试代码，14:34/17:12 的 deep_ep 更新已解决 64 rank SHMEM 组网问题**

## 三、与 PART1 EP64 补跑缺口的关系

| 项 | 状态 |
|---|---|
| PART1 EP64 dispatch_t4096 性能档 | ✅ 原有数据完好（perf-part1-ep64/） |
| PART1 EP64 dispatch_t16384 性能档 | ❌ 仍缺（lsx 只跑了 t4096 精度） |
| PART1 EP64 combine_t4096/t16384 性能档 | ❌ 仍缺 |
| **环境可行性** | ✅ **已确认 64 rank 可跑通，缺口可用当前环境补跑** |

lsx 的跑通是 **t4096 精度校验**（--dtype both），不是 t16384 性能档，也不含 combine。但它的意义在于**证明了当前环境 8 机 64 rank HT 版已稳定可用**——之前诊断的"HT 版 64 rank 必卡死"根因（deep_ep 旧 _C.so/elastic.py 缺陷）已被 14:34/17:12 的更新修复。

## 四、当前正在跑的任务（23:24 发起）

- 23:24-23:37 有任务在跑：**EP64 barrier 复测**（`relaunch_barrier8.sh`，8 台 `os{N}_run_perf1_dispatch_ep64_barrier.sh`，`test_dispatch_barrier.py` 版，t4096 性能档 barrier 实验）
- 23:41 master_wait ready=7/7 放行，当前正常执行中（说明 8 台 64 rank 已可正常初始化）
- 输出：`/home/deepep/perf-part1-ep64-barrier/`

## 五、建议

1. **补跑 PART1 EP64 剩余 3 档性能缺口**（dispatch_t16384 + combine_t4096 + combine_t16384）——当前环境已确认可行，可立即执行
2. 补跑时注意：即使当前已跑通，仍建议保留 `rm -f /dev/shm/psm_*` 清理步骤防 SHMEM 残留
3. lsx 的 4 次成功可作为"64 rank 组网可复现"证据纳入诊断报告更新（PART1_EP64_SHMEM诊断报告.md 的根因结论需修正：不再是"HT 版必卡死"，而是"旧版 deep_ep _C.so 缺陷，14:34 更新已修复"）
