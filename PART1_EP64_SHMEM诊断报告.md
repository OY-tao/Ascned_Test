# PART1 EP64 SHMEM status=-3 诊断报告

> 版本：v5（2026-10-26 14:05，v4.3 基础上补充**消除版本 warning 实测**——`_build_info.json` 对齐（期望改 7e36a92d）后 LL+hccl 重跑 EP32 dispatch 24 case：fix vs base ±1.6% 内无变化、无 `differs from` warning；dispatch 高 BS 劣化 vs 归档基线依旧复现（+9.7%~+24.6%）→ 版本不匹配 warning 与性能/劣化均无关）
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

### 2.7 SHMEM 库版本核对（已实测，10-26 11:00）

失败日志中每 rank 首次加载均出现：

```
deep_ep/_native.py:29: RuntimeWarning: Loaded SHMEM library differs from the SDK used to build this wheel
```

**10-26 已对该 warning 做了全量核实**，结论：**库版本差异是长期编译产物漂移，不是 10-25 分水岭原因**：

| 核对项 | 结果 |
|---|---|
| 8 台机器实际加载的库 | 完全一致：`libshmem.so` SHA=`7e36a92d...`，3903952 B，mtime **Oct 22 11:04**（install/ 与 build/ 两份一致） |
| wheel `_build_info.json` 期望 | SHA=`e6346c28...`（shmem revision `4dd80e21`，2026-10-07） |
| 源码 revision | 本地 shmem checkout = `4dd80e21`（与 wheel 期望**完全一致**）；`include/shmem.h` SHA=`57f8808b...` 也一致 |
| 差异定位 | 同 revision 源码，**本地 Oct 22 11:04 重编译产物（7e36a92d）≠ wheel 构建时 SDK 产物（e6346c28）**——编译环境/工具链差异所致 |
| 全盘是否有 e6346c28 | **无**（os1 全盘搜索 0 匹配，唯一候选已排除） |
| 10-24 成功 vs 10-26 失败 | **用的同一个 7e36a92d 库**（10-24 成功案例日志同样有该 warning）→ **库从未被替换** |

**结论**：该 warning 自 10-22 编译以来就存在，10-24 成功案例与 10-26 失败案例加载同一库文件，**排除"10-25 前后有人更新库导致行为变化"**。库版本核对不是当前瓶颈。顺带发现：os1 上存在 ≥40 种不同 SHA 的 libshmem.so（各用户/各任务各自编译的版本，如 zzh `338d2524`、b00943526 `77907fc7`、xie `4f861481` 等，互不干扰，各自用自己 LD_LIBRARY_PATH）。

### 2.8 实际加载库的来源链路（已实测，10-26 11:20）

**问：8 台机器实际加载的 `7e36a92d` 库是哪来的？** 完整链路：

1. **wheel 本身不带库**：`ascend_deepep-0.1.0.dev5-cp312`（Oct 23 17:12 正式 wheel 安装，无 direct_url）内部**不含 libshmem.so**——`pip show -f` 文件清单与 `RECORD` 均无 shmem 条目，site-packages/deep_ep/ 下只有 `_C.cpython-312-...so` + `libdeepep_kernels.so`
2. **动态链接器查找顺序**：`_C.so` 的 RPATH=`$ORIGIN`（`readelf -d` 证实），NEEDED=`libshmem.so` → 链接器**先在 deep_ep 包目录找** → 没有 → **回落 LD_LIBRARY_PATH**
3. **LD_LIBRARY_PATH 注入点**：`rerun_p1_ep64_os1.sh:8`（及 os2-8、rerun_p2_ep64_*.sh 同款）`export LD_LIBRARY_PATH="/home/y00889330/deepep/third-party/shmem/install/shmem/lib:..."` → **实际加载 install/shmem/lib/libshmem.so（7e36a92d）**
4. **该库的构建来源**：Oct 22 11:04 在 `/home/y00889330/deepep/third-party/shmem/` 本地 CMake 构建（CMakeCache：`CMAKE_CXX_COMPILER=/home/pkg/b070/cann-9.2.0/bin/bisheng`、RELEASE/-O3），`install_manifest.txt` 记录从 `build/lib/` 安装 4 个库到 `install/shmem/lib/`；`deepep-sdk.json`（同目录）记录 revision `4dd80e21`、CANN 9.2.0、target Ascend950、`source_patches: []`；构建后 2 分钟（11:06）生成 `ascend_deepep.egg-info`——**root 用户执行的 SDK 重建 + 安装**
5. **8 台一致**：os1-8 的 `install/shmem/lib/libshmem.so` 全部 mtime `Oct 22 11:04`、SHA `7e36a92d` 一致（构建后统一分发/同步）

**一句话**：实际加载的库 = **Oct 22 本地用 CANN 9.2.0 + bisheng 编译器从 shmem 源码（revision 4dd80e21）重新编译并 install 的 SDK**，经各 rerun 脚本 LD_LIBRARY_PATH 注入；wheel 不参与提供该库（wheel 构建时用的 SDK 是另一个编译产物 e6346c28，与本地产物 SHA 不同）。

### 2.9 shmem 版本会影响性能吗？（实测结论：**不影响稳态性能，只影响初始化成败**）

> **决定性证据（10-26 11:35 补充）：归档基线本身就是用 `7e36a92d` 这个库测出来的。**

| 维度 | 证据 |
|---|---|
| **归档基线与当前同库** | `archive/run_100case_os1.sh:9`、`run_dual_node_os1.sh:8`、`run_dispatch.sh:7`（Oct 22 基线采集脚本）的 LD_LIBRARY_PATH 与当前补跑脚本**完全相同**：`/home/y00889330/deepep/third-party/shmem/install/shmem/lib` → **官方归档基线（EP性能测试归档.xlsx）就是用 7e36a92d 测的**；当前所有不达标/劣化档位也跑在**同一个库**上——基线 vs 实测不存在"库版本不同"的系统性偏差 |
| **同库跑出达标性能** | `7e36a92d` 库（8 台统一）：PART1 dispatch_t4096 达标（10-24）、PART2 LL 96/96 档全齐（EP32/64 全矩阵）、combine 各组合全线吻合归档（≤1.9%）——**同一个库同时产出过达标与劣化数据，达标/劣化差异不可能是库版本造成** |
| **不同库同样达标** | b00943526（PART4）`77907fc7`：combine 全线 102~103%（393.8/396.7 GB/s），超归档基线——**另一版本库性能同样达标** |
| **两库二进制差异** | `7e36a92d`（3903952B，b070/cann-9.2.0）vs `77907fc7`（3903440B，b080/cann-9.2.0）：`cmp -l` 差异 **2717346 字节**（3.9MB 文件几乎全不同，属不同构建产物）——**但性能均达标，证明稳态性能对库版本不敏感** |
| **版本 vs 成败** | 64 rank 失败（HT 版）与成功（LL 版 k3retest）**用同一个 7e36a92d 库**；差异在 bootstrap 层（HT 默认 SHMEM bootstrap 卡死 ↔ LL 用 hccl/gloo bootstrap 成功），`_runtime.py:145` backend 仅决定进程组 allgather 走 gloo/hccl，**数据平面始终加载 libshmem.so** |
| **性能劣化≠库版本** | PART2 高 BS 劣化（t48 起 +5.7~23.5%）为系统性扩展开销，8 机独立复测偏差 <0.7% 确认稳定真实，与库版本无关（同库低 BS 段完全吻合基线 ≤4%） |
| **消除版本 warning 实测（10-26 13:21 新实验）** | `_build_info.json` 更新为期望 `7e36a92d`（对齐运行时库，**库文件未替换**，仍是 Oct 22 的 build/install 产物）后，LL+hccl 同口径重跑 EP32 dispatch 24 case（`perf-ep32-dispatch-shmemfix-20261026/`）：**24/24 全部 completed，无 `differs from` warning**；fix vs base（`/home/y00889330/perf-ep32-dispatch/` 旧提取）**24 档 delta 全部在 ±1.6% 内**（最大 +1.44%/-4.77%，k3_t64 档 -4.77% 为正向改善）——**消除版本不匹配 warning 前后性能完全一致，证明该 warning 与性能无关**；combine fix 脚本已备（`os1_run_perf_combine_ep32_fix20261026.sh`）但未跑 |

**关于"dispatch 不劣化"的准确解读（10-26 13:25）**：
- **修复后 vs 修复前（同口径 LL+hccl）**：24 档基本一致（±1.6%）→ **无额外劣化，dispatch 修复前后性能稳定** ✅
- **修复后 vs 归档基线（高 BS 档）**：t48 +14.3% / t64 +18.9% / t96 +24.6% / t128 +15.7% / t192 +9.7% / t256 +10.7% —— **劣化依然存在**（与修复前 +9.8%~+23.5% 几乎相同）→ 高 BS 劣化是 LL+hccl 口径固有现象，**与 SHMEM 版本不匹配无关**

**结论**：
1. **shmem 库版本不影响稳态性能**——不同编译产物（7e36a92d / 77907fc7 等）都跑出过达标甚至超基线数据；性能差异主要来自测试口径（LL/HT）、dtype（bf16/fp8）、数据规模与扩展开销，而非库版本。
2. **"性能不达标"与库版本无关（本次核实重点）**——归档基线（EP性能测试归档.xlsx）与当前所有补跑/复测数据**跑在同一个 7e36a92d 库上**：达标档（combine 全线 ≤1.9% 吻合、dispatch 低 BS ≤3.7%）和不达标档（dispatch 高 BS +5.7~23.5%、PART1 EP64 三档缺失）都是同一库产出。**同一库内部同时存在达标与不达标档位 → 不达标的原因只能是测试路径/规模/组网，不是库版本。**
3. **消除版本 warning 前后性能无变化（10-26 新实验佐证）**：`_build_info.json` 对齐（期望改 7e36a92d）后 LL+hccl 重跑 EP32 dispatch 24 case，与修复前同口径数据一致（±1.6%）；dispatch 高 BS 劣化 vs 归档基线依旧复现（+9.7%~+24.6%）——**版本不匹配 warning 与性能/劣化均无关**。
4. **库版本可能影响初始化稳定性**（理论上不同编译的 bootstrap 时序不同），但本环境 8 台同库、成败对照（HT vs LL 同库）已证明当前 64 rank 卡死**不是库版本引起**，而是 HT 版 bootstrap 组网方式缺陷。
5. **真正值得警惕的不是版本漂移，而是"换库"动作本身**：若有人把不同编译的 libshmem.so 覆盖进 8 台统一路径，可能导致跨节点行为不一致（部分节点旧库/新库混用），那才会引入新的不稳定——但目前 8 台 SHA 一致，无此风险。

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
4. **【已排除】SHMEM 库版本不一致**（2.7 节已实测）：8 台加载同一 `7e36a92d` 库（Oct 22 编译至今未变），10-24 成功与 10-26 失败用同一库；warning 是常驻编译产物漂移，**与 64 rank 卡死无关**
5. **【可选验证】调大 300s 超时**：`test_dispatch.py`/`test_combine.py` 的 `ElasticBuffer()` 调用加 `num_cpu_timeout_secs=900`（见 2.6 节）。预期**大概率无效**（成功案例 89s 完成初始化、2GB buffer 也失败、hccl 对照组秒级成功），但可低成本排除"恰好差一点"的可能

## 六、当前状态

- 8 台全空、无残留进程；`/home/y00889330/perf-part1-ep64-rerun/` 下 3 档失败目录保留（可复看日志）
- dispatch_t4096 成功数据保留（node0 目录 + os1.log 85978B）
- PART1 EP64 3 档缺口仍在；修复方向见第五节，**改 hccl 后即可重试**
- 各节点 `.sync/` 残留 waiting/go 标记（下一轮 run_case 会 rm 重建，无碍）
- 各节点 `.sync/` 残留 waiting/go 标记（下一轮 run_case 会 rm 重建，无碍）
