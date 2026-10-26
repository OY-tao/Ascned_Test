# PART1 EP64 SHMEM status=-3 诊断报告

> 时间：2026-10-25 19:00-19:20
> 对象：PART1 EP64 缺失 3 档（dispatch_t16384 / combine_t4096 / combine_t16384）重试全部失败，报错 `aclshmemx_init_attr_with_buffers failed (status=-3)`
> 结论：**根因已定位——os6 在三轮重跑中全程缺席（进程从未被拉起），64 进程 SHMEM 组网只有 7 台参与，allgather 永远等不到第 8 台，300s 超时 → status=-3**

---

## 一、现象复述

18:29-18:55 三轮重跑（脚本 `rerun_p1_ep64_osN.sh`，RERUN_ONLY=dispatch:16384,combine:4096,combine:16384）全部失败，失败进程报错一致：

```
RuntimeError: aclshmemx_init_attr_with_buffers failed (status=-3)
```

对应 aclshmem 日志（os1 失败进程）：

```
[SHM_SHMEM store_tcp_config.cpp:36]   WaitFinished timeout after 300000ms, link may be broken
[SHM_SHMEM store_tcp_config.cpp:290]  send get for key: SHM_(0)_S_0_1_GW, get null response
[SHM_SHMEM store_net_group_engine.cpp:273] store get key: SHM_(0)_S_0_1_GW failed, result:socket error
[SHM_SHMEM shmemi_bootstrap_config_store.cpp:178] Group AllGather timeout or store failure
[SHM_SHMEM shmem_init.cpp:1252] return shmem error: -3 - g_boot_handle.allgather(...) failed
```

**关键特征**：全部是 **AllGather timeout（300000ms=5分钟）**，不是端口占用/资源不足类报错。

## 二、决定性证据（os6 缺席）

| 证据 | 内容 |
|---|---|
| os6 stdout | 停留在**昨天 10-24 20:55** 的旧运行（`[20:55:02]`、`[21:00:00]`），今天 18:33 下发三轮无任何执行记录 |
| os6 输出目录 | `perf1_combine_t16384...` 目录 mtime = **10-24 20:27**（旧数据）；其余 7 台均为 **10-25 18:49**（新数据） |
| os6 aclshmem 日志 | 无任何 18:33-19:00 时段的日志（最新停在 12:51） |
| os6 18:33-19:00 新文件 | 仅 `.sync/go`（18:49，同步标记），无任何脚本运行产物 |
| os6 node6 目录 | 三轮均无新 node6 目录产出 |
| 对照：昨天成功案例 | dispatch_t4096 的 `.os6.log` = 10-24 20:13（os6 参与，成功） |
| 对照：今天成功案例 | k3retest（18:07-18:27）os6 stdout 明确显示 5 档全参与、全成功 |

## 三、失败机制推演

1. 启动器 `/tmp/launch_p1_ep64.sh` 对 os6 走 os2 跳板下发（`ssh root@177.22.22.1 "nohup bash -c '... ssh root@177.22.26.1 nohup bash rerun_p1_ep64_os6.sh ...'"`）
2. os6 的脚本文件在 18:33 已同步（文件 mtime=18:33），但**进程从未真正启动**（无 stdout 更新、无日志、无 node 目录）
3. os1 的 `master_wait()` 只等 **1 台** waiting 就放行（`ready -lt 1`，而 k3retest 等满 7 台 `ready -lt ${#SLAVE_HOSTS[@]}`）
4. 每次只有 7 台（os1-5, os7-8）进入 SHMEM 初始化；64 进程的组 allgather 等不到 os6 的 8 个 rank
5. 5 分钟超时 → status=-3（AllGather timeout），三轮全灭

**这也解释了为什么 dispatch_t4096 昨天成功、今天 18:37-18:55 三次全失败**：昨天 os6 参与（20:11-20:13 日志），今天 os6 缺席。

## 四、为什么 os6 没被拉起（待进一步验证）

- launch 脚本跳板链：`本地 → os1 → (os6 分支) ssh root@177.22.22.1 → ssh root@177.22.26.1 nohup bash ...`
- os2→os6 免密实测连通正常（JUMP-OS2-OK），排除免密失效
- 最可能：跳板嵌套 + nohup 后台启动在 ssh 会话退出后被回收（与记忆中"本地 Bash 工具对 ssh 远程 nohup 会 SIGTERM"同类问题；os6 因为是**二级跳板**，nohup 进程的父链更脆弱，首个子 shell 退出即被杀）
- 需要实际重测一次 os6 单独下发来定位断链点

## 五、修复建议（按优先级）

1. **【首选】os6 改用 os1 跳板**（实测 os1→os6 直连正常），启动脚本把 os6 分支的跳板从 os2 改为 os1，或写成独立脚本文件在 os6 上一次性执行
2. **【稳健】主控同步改为等满 7 台**：把 `rerun_p1_ep64_os1.sh` 的 `master_wait()` 从 `ready -lt 1` 改为 `ready -lt ${#SLAVE_HOSTS[@]}`（对齐 k3retest），可防"某台没起来还硬跑"的假成功
3. **【兜底】给 slave_wait 加超时**：对齐 k3retest 的 600s 超时逻辑，避免永久挂起
4. **【排查】os6 单独下发一次**（不起 7 台），确认 nohup 断链点，作为长期修复依据

## 六、当前状态

- 8 台全空、无残留进程、端口干净（19090/19091 无监听）
- dispatch_t4096 成功数据保留（node0 目录）
- PART1 EP64 3 档缺口仍在，修复启动链路后即可重试
- 各节点 `.sync` 残留 waiting/go 标记（下一轮 run_case 会 rm 重建，无碍）
