# PART1 EP64 SHMEM status=-3 诊断报告

> 版本：v3（2026-10-26 10:40，v2 基础上补充 300s 超时来源分析——新增 2.6/2.7 节，结论不变）
> 对象：PART1 EP64 缺失 3 档（dispatch_t16384 / combine_t4096 / combine_t16384）多轮重试全部失败，报错 `aclshmemx_init_attr_with_buffers failed (status=-3)`
> 结论（v2 修正）：**根因 = HT 版（test_dispatch.py / test_combine.py）在 8 机 64 rank 规模下 SHMEM 组网 AllGather 必然卡死（300s 超时 → status=-3），与节点是否缺席无关**。10-26 凌晨修复版已排除"os6 缺席 / 同步等台数不足"两个旧嫌疑，8 台全部真实参与仍全灭；对照 LL 版（test_ep_low_latency.py + `--group-backend hccl`）同规模 8 机稳定成功。

---

## 〇、日志路径速查表（排查必读）

所有路径均以对应节点为准（`/home/y00889330` 为各节点**本地磁盘**，非共享存储，8 台各有一份）：

| 内容 | 路径（按节点） |
|---|---|
| **主控 stdout（os1）** | `/home/y00889330/rerun/rerun_p1_ep64_os1.stdout` |
| **各节点脚本 stdout** | `/home/y00889330/rerun/rerun_p1_ep64_os${N}.stdout`（os1~os8，os6 在其本地） |
| **各节点运行日志（osN.log）** | `/home/y00889330/perf-part1-ep64-rerun/perf1_{op}_t{TOK}_h7168_k6_e384.os${N}.log`（每节点本地各一份） |
| **结果/存档目录（各节点本地）** | `/home/y00889330/perf-part1-ep64-rerun/perf1_{op}_t{TOK}_h7168_k6_e384/`（内含 `node${N}/`） |
| **SHMEM 同步标记** | `/home/y00889330/perf-part1-ep64-rerun/.sync/`（waiting_$RUN / go_$RUN） |
| **aclshmem 底层日志（各节点本地）** | `/root/shmem/log/aclshmem_{PID}_{时间戳}.log`（8 台各自 /root/shmem/log/） |
| **启动器（下发脚本）** | `/tmp/launch_p1_ep64.sh`（10-26 修复版，os6 走 os1 跳板） |
| **启动记录（relay）** | `/home/y00889330/rerun/relay_p1_ep64.log` |
| **脚本本体** | `/home/y00889330/rerun/rerun_p1_ep64_os${N}.sh` |
| **旧版脚本备份** | `/home/y00889330/rerun/rerun_p1_ep64_os${N}.sh.bak_1025`（10-25 版，等 1 台即放行） |
| **成功对照（LL+hccl，k3retest）** | `/home/y00889330/rerun/rerun_p2_ep64_k3retest_os${N}.sh`、`/home/y00889330/perf-ep64-dispatch-k3retest/` |

---

## 一、现象复述

### 1.1 首轮（10-25 18:29-18:55，v1 报告已述）

三轮重跑（`RERUN_ONLY=dispatch:16384,combine:4096,combine:16384`，旧版脚本 `.bak_1025`）全部失败，失败进程报错一致：

```
RuntimeError: aclshmemx_init_attr_with_buffers failed (status=-3)
```

对应 aclshmem 日志（os1 失败进程，`/root/shmem/log/aclshmem_*.log`）：

```
[SHM_SHMEM store_tcp_config.cpp:36]   WaitFinished timeout after 300000ms, link may be broken
[SHM_SHMEM store_tcp_config.cpp:290]  send get for key: SHM_(0)_S_0_1_GW, get null response
[SHM_SHMEM store_net_group_engine.cpp:273] store get key: SHM_(0)_S_0_1_GW failed, result:socket error
[SHM_SHMEM shmemi_bootstrap_config_store.cpp:178] Group AllGather timeout or store failure
[SHM_SHMEM shmem_init.cpp:1252]  return shmem error: -3 - g_boot_handle.allgather(...) failed
```

**关键特征**：全部是 **AllGather timeout（300000ms=5 分钟）**，不是端口占用/资源不足类报错。

### 1.2 v1 当时判定（已被 v2 推翻）

v1 基于首轮证据判定"os6 缺席"：os6 stdout 停留在 10-24 旧运行、无新日志、node 目录无产出，而 7 台有数据。据此归因"64 进程组网只有 7 台 → allgather 等不到第 8 台 → 超时"。

---

## 二、10-26 凌晨修复版重试（决定性证据，v2 新增）

针对 v1 两个嫌疑（os6 缺席、master_wait 只等 1 台），00:28 修复后重试：

### 2.1 修复内容（diff `.sh` vs `.sh.bak_1025`）

| 项 | 旧版（10-25） | 修复版（10-26） |
|---|---|---|
| master_wait | 等 1 台即放行（`ready -lt 1`） | 等满 7 台（`ready -lt ${#SLAVE_HOSTS[@]}`）+ 200 次超时兜底 |
| 同步标记 | 全局 `waiting`/`go`（跨档串扰） | 每档独立 `waiting_$RUN`/`go_$RUN` |
| os6 跳板 | 经 os2（二级嵌套，易断） | **经 os1 直连跳板**（`launch_p1_ep64.sh` 专门分支） |

### 2.2 下发成功记录（`/home/y00889330/rerun/relay_p1_ep64.log`）

```
===== [00:28:44] 启动 PART1 EP64 缺失3档 (os6 走 os1 跳板) =====
os1 started
os6 started (via os1)      ← os6 经 os1 跳板成功下发
os2 started
...
===== [00:28:55] PART1 EP64 8 台全部下发 =====
```

### 2.3 主控放行记录（`/home/y00889330/rerun/rerun_p1_ep64_os1.stdout`）

```
[00:55:59] master_wait ready=7/7, 放行 perf1_dispatch_t16384_h7168_k6_e384   ← 等满 7 台
[01:01:51] os1 perf1_dispatch_t16384_h7168_k6_e384 退出码=1                 ← 仍失败
[01:02:20] master_wait ready=7/7, 放行 perf1_combine_t4096_h7168_k6_e384
[01:08:11] os1 perf1_combine_t4096_h7168_k6_e384 退出码=1
[01:08:45] master_wait ready=7/7, 放行 perf1_combine_t16384_h7168_k6_e384
[01:09:22] os1 启动: perf1_combine_t16384_h7168_k6_e384
```

### 2.4 os6 真实参与的铁证（此前唯一盲区）

| 证据 | 内容 |
|---|---|
| os6 本地运行日志 | `/home/y00889330/perf-part1-ep64-rerun/perf1_dispatch_t16384_h7168_k6_e384.os6.log`（7910B）、`...combine_t4096...os6.log`（9839B）、`...combine_t16384...os6.log`（9723B），**均 10-26 01:02~01:15 时间戳** |
| os6 本地 aclshmem 日志 | `/root/shmem/log/aclshmem_{2276743..2276750}_20261026010200.log`（combine_t4096，8 个 rank 全在）、`aclshmem_{2291866..}_20261026011458.log`（combine_t16384） |
| os6 报错内容 | 与 os1 完全一致：`WaitFinished timeout after 300000ms` → `status=-3` |
| 8 台日志清单 | os2/3/4/5/7/8 本地均有三档 `.os${N}.log`（9000~10000B，10-26 时间戳）——**8 台全参与** |

### 2.5 超时时间线（每档均卡满 300s）

| 档位 | 启动时间（放行） | 超时时间（aclshmem 日志） | 等待 |
|---|---|---|---|
| dispatch_t16384 | 00:56:46 | 01:01:46（os1 `aclshmem_713106_20261026010146.log`） | 300s |
| combine_t4096 | 00:57:00 | 01:02:00（os6 `aclshmem_2276743_20261026010200.log`） | 300s |
| combine_t16384 | 01:09:58 | 01:14:58（os6 `aclshmem_2291866_20261026011458.log`） | 300s |

每档都是**8 台 64 rank 全部拉起、group allgather 等不到全部对端、卡满 300s 超时 → status=-3**。

### 2.6 300s 超时来源分析（是否可调大？）

**结论：300s 超时来自 deep_ep Python 层 `ElasticBuffer` 的 `num_cpu_timeout_secs=300` 默认值，理论可调大，但调大大概率无效（不是慢，是死等）。**

证据链（源码追踪）：

```
test_dispatch.py:1644  ElasticBuffer(rank, world_size, num_bytes, num_max_tokens_per_rank, hidden, num_topk, explicitly_destroy=True)
  → 未传 num_cpu_timeout_secs → 用默认值 300
deep_ep/buffers/elastic.py:182  num_cpu_timeout_secs: int = 300     ← 300s 源头
deep_ep/buffers/elastic.py:239  self._runtime = Runtime(group, num_bytes, num_cpu_timeout_secs, ...)
deep_ep/_runtime.py:75         timeout=120  ← Runtime 默认 120，但被 num_cpu_timeout_secs=300 覆盖
deep_ep/_runtime.py:233        native.Runtime(rank, size, capacity, address, timeout=300, ...)
  → aclshmem 内部 ClientWaitContext(300000ms) → WaitFinished timeout after 300000ms
```

- aclshmem 库自身默认值 `DEFAULT_TIMEOUT = 120`（`third-party/shmem/include/host/shmem_host_def.h:32`）、`SMEM_DEFAUT_WAIT_TIME = 120`（`store_utils.h:48`），**但被 Python 层传入的 300 覆盖**
- `ElasticBuffer` 校验（elastic.py:217）只限制 `num_device_timeout_secs != 100` 才报错，**`num_cpu_timeout_secs` 不校验，可传任意值**

**为何调大无效（核心判据）**：
1. **buffer.init 耗时对比**——成功案例 dispatch_t4096（10-24 20:12）`buffer.init` 仅 **89 秒**（20:10:54 → 20:12:23）；失败案例等 300s 连一半 rank 的 `buffer.init.done` 都等不到。64 rank allgather 正常时 89s 完成，说明 300s 余量非常充足，**不是"刚好差一点"**
2. **combine_t4096 仅 2.08 GiB 也失败**（与成功案例 t4096 2.45 GiB 同量级）——排除"大 buffer 初始化慢"论，纯组网问题
3. **对照 LL 版 hccl 组网秒级成功**——SHMEM 组网在此规模存在结构性缺陷，非速度问题

**若仍想验证"调大超时"**（排除万一），改动点：`test_dispatch.py:1644` 的 `ElasticBuffer(...)` 调用加 `num_cpu_timeout_secs=900`，`test_combine.py` 同理；预计 10 分钟内可验证一档（等 900s 超时）。

### 2.7 重要环境线索（新增，可能与 300s 无关但值得追）

失败日志中每 rank 首次加载均出现：

```
deep_ep/_native.py:29: RuntimeWarning: Loaded SHMEM library differs from the SDK used to build this wheel
```

- 该 warning **成功案例（dispatch_t4096）同样存在**（18/21/24/27/30 行），说明 SHMEM 库版本不一致是**常驻状态**，非失败充分条件
- `_native.py` 校验：运行时加载的 `/home/y00889330/deepep/third-party/shmem/install/shmem/lib/libshmem.so`（经 LD_LIBRARY_PATH 注入）SHA256 ≠ wheel `_build_info.json` 记录值
- **值得追查**：10-25 之后是否有人更新过该 libshmem.so 或 SDK，导致 64 rank 组网行为变化（10-24 能成功 ↔ 10-25 后必失败的分水岭）

---

## 三、根因结论（v2 修正）

**HT 版（test_dispatch.py / test_combine.py）在这套 8 机环境的 64 rank SHMEM 初始化本身不稳**，证据链：

1. **节点缺席论排除**：os6 经 os1 跳板成功参与、8 台全有日志、同步等满 7/7，仍全灭；
2. **同步逻辑排除**：修复版 master_wait 等满 7 台 + 每档独立标记 + 超时兜底，全部生效（stdout 明确 ready=7/7），仍全灭；
3. **大 buffer 论排除**：combine_t16384（2.2GB 通讯 buffer）与 dispatch_t16384、combine_t4096 同模式失败——不是"buffer 太大触发"；
4. **对照组（关键）**：同日同 8 机、**LL 版 test_ep_low_latency.py + `--group-backend hccl`**（k3retest，`rerun_p2_ep64_k3retest_os1.sh`）5 档（t48~t192）全部稳定成功——**同规模 64 rank 用 hccl 组网可正常初始化**；
5. **接口差异**：`test_dispatch.py` 无 `--group-backend` 参数（仅 `--shmem-ip-port`，默认 SHMEM 组网），无法像 LL 版那样切 hccl；即使设了 `DEEPEP_SHMEM_ENDPOINT` 环境变量（`tcp://177.22.21.1:19091`），SHMEM 组网在 64 rank 下仍卡 allgather。

**一句话**：HT 版在这套环境的 64 rank 下走 SHMEM 组网必然 300s 超时失败，属于 HT 脚本/组网适配问题，不是运行环境、节点或脚本同步 bug。

---

## 四、为什么 dispatch_t4096 能成功（唯一幸存档）

| 项 | dispatch_t4096（成功） | 三档（失败） |
|---|---|---|
| 时间 | 10-24 20:11-20:13（首跑） | 10-25 18:37+ / 10-26 00:56+（重试） |
| 规模 | 64 rank 同规格 | 64 rank 同规格 |
| 结果 | ✅ 成功（`perf1_dispatch_t4096_h7168_k6_e384.os1.log` 85978B，node0 数据完好） | ❌ status=-3 |

同一规格 10-24 能成功、10-25 之后必失败 → **失败具有时间/环境相关性**（可能与该时点后的 SHMEM 组网/驱动状态变化有关），但 10-26 修复版排除了脚本与节点因素后仍必失败，说明当前状态下 HT 版 64 rank 已不可复现成功。**dispatch_t4096 数据保留作基线**（该档已达标）。

---

## 五、修复方向（按优先级）

1. **【首选】HT 版也改走 hccl 组网**：给 `test_dispatch.py` / `test_combine.py` 增加 `--group-backend hccl` 透传（或改默认值），对齐 LL 版 k3retest 已验证的成功路径；这是最可能根治的方案
2. **【验证】单档小规模对照**：先用 4 机 32 rank（EP32 已在 PART2 验证稳定）跑 HT 版，若 32 rank 成功、64 rank 失败，进一步坐实"64 rank SHMEM 组网上限"
3. **【兜底】改用 LL 版补 PART1 口径**：若 HT 版无法修复，可评估以 LL 版（已稳定）按 test_shape 口径补 3 档数据（需确认口径可对齐）
4. **【排查】SHMEM 库版本不一致**（2.7 节）：比对运行时 `libshmem.so` 与 wheel `_build_info.json` 期望 SHA256，确认 10-25 前后是否有 SDK/库更新导致 64 rank 组网行为变化；若确认库被换，重装匹配版本可能直接解决
5. **【可选验证】调大 300s 超时**：`test_dispatch.py`/`test_combine.py` 的 `ElasticBuffer()` 调用加 `num_cpu_timeout_secs=900`（见 2.6 节）。预期**大概率无效**（成功案例 89s 完成初始化、2GB buffer 也失败、hccl 对照组秒级成功），但可低成本排除"恰好差一点"的可能

## 六、当前状态

- 8 台全空、无残留进程；`/home/y00889330/perf-part1-ep64-rerun/` 下 3 档失败目录保留（可复看日志）
- dispatch_t4096 成功数据保留（node0 目录 + os1.log 85978B）
- PART1 EP64 3 档缺口仍在；修复方向见第五节，**改 hccl 后即可重试**
- 各节点 `.sync/` 残留 waiting/go 标记（下一轮 run_case 会 rm 重建，无碍）
- 各节点 `.sync/` 残留 waiting/go 标记（下一轮 run_case 会 rm 重建，无碍）
