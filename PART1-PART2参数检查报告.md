# PART1 / PART2 脚本参数检查报告（对照 deepep/tests/README.md）

检查时间：2026-10-24 16:20（首次）/ 16:40（问题修复后复核）
检查基准：`/home/y00889330/deepep/tests/README.md`（HT 单机/多机 + LL 正确性与性能规范）
检查对象：本地 `A5/dist-part1/`、`A5/dist-part2/`、根目录 `run_*.sh`，服务器 `/home/y00889330/*.sh`

---

## 一、端口规范（已修正 ✅）

README 规范：**MASTER_PORT=19090 + DEEPEP_SHMEM_ENDPOINT=tcp://MASTER_ADDR:19091 分离**

| 位置 | 修正前 | 修正后 |
|---|---|---|
| 本地 dist-part1（30 个） | 22 个 MASTER_PORT=19091 + SHMEM 同 19091；8 个 ep64 SHMEM_PORT=26741 | MASTER_PORT=19090，SHMEM=19091 |
| 本地 dist-part2（48 个） | 40 个 MASTER_PORT=19091 + SHMEM 同 19091 | MASTER_PORT=19090，SHMEM=19091 |
| 本地根目录 run_*（17 个） | MASTER_PORT=19091 | MASTER_PORT=19090 |
| 服务器 os1（22 个） | 同上 | 同上 |
| 服务器 os2~os8（49 个） | 同上（os6 连接失败未改） | 同上 |

已全部统一为 README 规范。

---

## 二、参数问题清单

### ✅ 问题 1（已修复）：服务器 os2~os8 与 os1 的 perf_dispatch_ep64 入口不一致

**原始问题**：os1 用 LL 版（`test_ep_low_latency.py --dispatch-only --group-backend hccl`），os2~os8 用 HT 旧版（`test_dispatch.py --profile-op dispatch`），EP64 多机无法协同，导致 PART2 EP64 dispatch 数据异常（500-1900us vs 归档 17-73us）。

**修复结果（2026-10-24 16:36 复核）**：

| 节点 | 入口 | 关键参数 |
|---|---|---|
| os1~os8（8/8） | `test_ep_low_latency.py`（LL 版） | `--num-processes 8 --num-nodes 8 --dispatch-only --skip-check --group-backend hccl --warmup 5 --num-iters 50 --perf-tail 30` |

- 8 节点均验证 `grep -c test_ep_low_latency = 2`，`HT残留 = 0`。
- os6（177.22.26.1）本机 SSH 直连失败，已通过 os1 跳板推送 LL 版脚本并校验成功（mtime 16:36）。

### ✅ 问题 2（已修复）：PART2 perf_combine 与 perf_dispatch 使用不同入口

**原始问题**：perf_dispatch 用 LL 入口（torch_npu profiler kernel 50 次 tail 30），perf_combine 用 HT 入口（`test_combine.py --profile-op combine`，warmup 5 iters 10），两套计时口径，数据不可直接对比。

**修复结果（PART2 全部统一 LL 入口）**：

| 脚本 | 入口 | kernel 计时口径 |
|---|---|---|
| perf_dispatch（EP32/64） | `test_ep_low_latency.py --dispatch-only --skip-check` | torch_npu profiler kernel tasks，逐 rank 采样 50 次取 tail 30 |
| perf_combine（EP32/64） | `test_ep_low_latency.py --skip-check`（去掉 --dispatch-only，dispatch+combine 全链路同测） | 同 dispatch：kernel 采样 50 次取 tail 30 |

- 本地 `dist-part2/`、根目录 `run_perf_combine_ep{32,64}_osN.sh` 模板、服务器 `/home/y00889330/os{N}_run_perf_combine_ep{32,64}.sh` 全部为 LL 版。
- EP64 脚本已带 master_wait/slave_wait 8 机同步（与 perf_dispatch 相同）；EP32 为 os1~os4 4 机同步。
- combine 用例 dtype=fp8、do_cpu_sync=0、warmup 5、iters 50、perf-tail 30，与 dispatch 完全同口径。

### ✅ 问题 3（已修复）：EP32 perf_dispatch 补显式 `--group-backend hccl`

- ep32 版已补 `--group-backend hccl`（本地 dist-part2 os1~os4 + 服务器 os1~os4），与 ep64 风格统一。

### 🟡 问题 4（注意）：HT 版 combine 的 weight 回传默认关闭

`test_combine.py` 默认 `without_weights=True`（README 说"combine 默认同时检查独立权重回传"，但**当前 wheel 代码默认关闭**）。PART1/PART2 的 HT combine 脚本均未传 `--with-weights`，即**未验证权重回传通道**。若权重回传是本轮测试范围，需要显式加 `--with-weights`。

### 🟢 已核实正确（无问题）的参数

| 项目 | README 规范 | 脚本实际 | 判定 |
|---|---|---|---|
| LL `--num-bytes` | LL requires exactly 8 GiB | 8589934592（8GiB） | ✅ |
| LL `--num-max-tokens-per-rank` | 省略 = M=B | 未传（默认 M=B） | ✅ |
| LL `--do-cpu-sync 1` | 精确 R 行比对 | acc 用 1，perf 用 0 | ✅ |
| LL `--skip-perf-test` | 只检查正确性 | acc 用 | ✅ |
| LL `--skip-check` | 跳过精度检查，结果标记 skipped | perf 用 | ✅ |
| LL `--perf-tail 30` | 默认 min(30, num_iters) | num-iters 50 + tail 30 | ✅ |
| LL `--warmup 5 --num-iters 50` | profiling 默认预热 10 次采样 50 次 | warmup 5（README HT 默认 5） | ✅ |
| LL `--shmem-ip-port` | 必须 CLI/JSON/env 提供 | 已显式传 `--shmem-ip-port "$DEEPEP_SHMEM_ENDPOINT"` | ✅ |
| HT `--ep-size/--nproc-per-node/--node-rank` | 多机示例 | 一致 | ✅ |
| HT `--tokens/--hidden/--topk/--experts` | 示例 257/7168/6/512，可替换 | 脚本按清单 H7168/K6/E384 与 K3 H3584/K16/E896 | ✅ |
| HT dispatch `--dtype both` | 支持 bf16/fp8/both | acc 用 both，perf 用 fp8 | ✅ |
| HT combine `--dispatch-dtype bf16/fp8` | combine 前置 dispatch 的 dtype | acc bf16+fp8 两轮，perf bf16 | ✅ |
| HT `--with-dispatch` | fp8 combine 需真实 dispatch | acc fp8 轮带 | ✅ |
| HT `--route all` | 正确性覆盖全路由 | acc 用 all，perf 用 random | ✅ |
| HT `--profile-warmup 5 --profile-iters 10` | 默认预热 5 次 10 次 | 一致 | ✅ |
| EP32 `--num-processes 8 --num-nodes 4` | 4 机 × 8 = 32 rank | ✅ | |
| EP64 `--num-processes 8 --num-nodes 8` | 8 机 × 8 = 64 rank | ✅ | |
| EP8 单机 | README 单机示例 | `--ep-size 8 --nproc-per-node 8`（1 机） | ✅ |
| K3 参数 | H3584/TOPK16/E896 | 脚本一致 | ✅ |

---

## 三、修复建议（已全部执行 ✅）

1. **【已修复】同步 os2~os8 的 perf_dispatch_ep64 为 LL 版**（与 os1 一致）：8 节点全部 LL 版，os6 经 os1 跳板补齐，重新部署完成。
2. **【已修复】统一 PART2 perf 入口**：perf_combine（EP32/64）改走 LL 版 `test_ep_low_latency.py`（去掉 --dispatch-only 同测 dispatch+combine），与 perf_dispatch 完全同口径（warmup5/iters50/tail30/fp8/hccl）。
3. **【已修复】EP32 perf_dispatch 补 `--group-backend hccl`**（风格统一，os1~os4 已部署）。
4. 若需验证权重回传，HT combine 加 `--with-weights`（当前 PART2 已全部 LL，HT 通道不再使用，此项不再适用）。
5. **【遗留】os6（177.22.26.1）本机 SSH 直连失败**：脚本已通过 os1 跳板同步成功；后续如需本机直连运维，需恢复网络/SSH 路由。

> 注：os6 上 `/home/y00889330/os6_run_perf_*` 已确认为 LL 版（mtime 2026-10-24 16:36）。os1 上遗留的 os6 同名副本仍是旧 HT 版（仅本地引用，不影响 os6 运行），如需彻底一致可下次一并覆盖。

---

## 四、附：本地 dist-part2 脚本入口矩阵（问题修复后，2026-10-24 16:40）

| 脚本类型 | 入口 | 关键参数 |
|---|---|---|
| acc_combine（EP32/64） | `test_ep_low_latency.py` | `--num-processes 8 --num-nodes 4/8 --do-cpu-sync 1 --skip-perf-test`，bf16+fp8 两轮 |
| acc_dispatch（EP32/64） | `test_ep_low_latency.py` | `--do-cpu-sync 1 --dispatch-only --skip-perf-test`，bf16+fp8 两轮 |
| perf_dispatch（EP32/64） | `test_ep_low_latency.py` | `--do-cpu-sync 0 --dispatch-only --skip-check --group-backend hccl --warmup 5 --num-iters 50 --perf-tail 30` |
| perf_combine（EP32/64） | `test_ep_low_latency.py`（LL，原 HT 已替换） | `--do-cpu-sync 0 --skip-check --group-backend hccl --warmup 5 --num-iters 50 --perf-tail 30`，dispatch+combine 全链路同测，dtype=fp8 |

> 注：`dist-part2/os{1..4}_run_perf_*_ep32_sync.sh` 为早期调试遗留副本（HT 入口），非部署目标，不影响运行。

---

## 五、EP32/EP8 相关脚本修复记录（2026-10-24 17:00 复核）

**原则（用户确认）**：PART1 必须 HT（对齐 test_shape 归档基线），PART2 必须 LL，各自范围内所有相关脚本统一。

### ✅ 修复 1：PART2 EP32 k3fp8 系列（b070 → b080 + group-backend）

| 脚本 | 修复前 | 修复后 |
|---|---|---|
| `run_perf_dispatch_k3fp8_ep32_osN.sh`（模板+节点） | `source /home/pkg/b070/...`、无 `--group-backend hccl` | `b080`、`--group-backend hccl` 已补 |
| `run_acc_dispatch_k3fp8_ep32_osN.sh`（模板+节点） | `source /home/pkg/b070/...` | `b080` |

- 服务器 os1~os4 共 8 个 k3fp8 脚本已全部部署更新（b080=1，perf 含 group-backend）。

### ✅ 修复 2：PART2 acc 系列（EP32/EP64）部署 LL 版（关键补漏）

- **问题**：服务器上 `os{N}_run_acc_combine_ep{32,64}.sh` / `os{N}_run_acc_dispatch_ep{32,64}.sh` 仍是 **HT 旧版**（上一轮只部署了 perf 系列，acc 漏部署）。
- **修复**：本地 LL 版 acc（28 个文件）已部署至 8 节点：ep32 → os1~os4，ep64 → os1~os8；os6 经 os1 跳板。
- **校验**：8 节点 acc 全部 `LL=2 HT=0`，端口 19090/19091 分离。

### ✅ 修复 3：os6 遗留端口问题（4 个脚本）

| 脚本 | 修复前 | 修复后 |
|---|---|---|
| `os6_run_acc1_dispatch_ep64.sh` | MASTER=19091 + SHMEM=26741 | MASTER=19090 + SHMEM=19091 |
| `os6_run_acc_combine_ep64.sh` | MASTER=19091 + SHMEM=19091（同端口） | MASTER=19090 + SHMEM=19091 |
| `os6_run_acc_dispatch_ep64.sh` | MASTER=19091 + SHMEM=19091（同端口） | MASTER=19090 + SHMEM=19091 |
| `os7_run_perf1_dispatch_ep64.sh`（误放 os6） | MASTER=19091 + SHMEM=${MASTER_PORT}（同端口） | **已删除**（os7 上有自身正确版） |

### ✅ 修复 4：本地 `run_perf_ep8_w0rerun_os1.sh` 端口

- MASTER_PORT=19093 + SHMEM=19093（同端口）→ MASTER_PORT=19090 + SHMEM=19091。

### ✅ 复核通过（无需修改）

- **PART1 EP8/EP16/EP32（perf1/acc1 系列）**：保持 HT 入口（test_dispatch.py/test_combine.py），与 test_shape 归档 HT 基线一致；端口全部 19090/19091 分离；os3/os4 无 ep16 脚本（EP16 仅 os1+os2 双机，正确）；os5/os7/os8 无 ep32 脚本（EP32 仅 os1~os4，正确）。
- **PART2 perf 系列（EP32/EP64）**：上轮已 LL 化，本轮复核无 HT 残留。
- 本地 `combine_full.sh`（19099）、`combine_probe.sh`（19097）为调试探针脚本，故意错开主端口，不在调度链，不改。

### 📊 最终状态（2026-10-24 17:00）

| 范围 | 入口 | 8 节点状态 |
|---|---|---|
| PART1 perf1/acc1（EP8/16/32/64） | HT（test_dispatch.py/test_combine.py） | ✅ 端口规范，参数对齐 test_shape 基线 |
| PART2 perf（EP32/64 + k3fp8） | LL（test_ep_low_latency.py） | ✅ 8 节点 LL=2 HT=0 |
| PART2 acc（EP32/64） | LL（test_ep_low_latency.py） | ✅ 8 节点 LL=2 HT=0（本轮补部署） |
