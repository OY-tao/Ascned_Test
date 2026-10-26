# EP8-64 性能与精度测试 Case 清单

> 依据：`C:/Users/OY/Desktop/AITest/EP性能测试归档.xlsx`（工作表「DeepEP低时延」）+ `test_shape.txt` 性能基线口径
> 生成时间：2026-10-23

## 〇、PART1 / PART2 / PART3 分工（2026-10-23 更新）

| 部分 | Case 来源 | 口径 | 脚本 |
|------|----------|------|------|
| **PART1** | `test_shape.txt` 性能基线网格（EP8/16/32/64 × T4096/16384，H7168 K6 E384，random，不传 weights） | 严格对齐基线表，dispatch FP8 / combine BF16 | `run_{perf1,acc1}_dispatch_ep{8,16,32,64}*.sh`（本轮新写） |
| **PART2** | `EP性能测试归档.xlsx` 算子表（D 组 / A1+B1 / A2+B2，低时延曲线） | 与基线同 profile 口径，但 shape 来自算子表 | `run_{perf,acc}_{dispatch,combine}_ep{8,32,64}*.sh`（已整理好） |
| **PART3** | MegaMoE 模式（`deep_ep_experimental.megamoe`，独立 wheel），高吞吐 EP32/64 + 泛化精度 EP8/16/32/64 | megamoe benchmark 口径（BS4096 H7168 K6 E384 S0，fp8） | `/mnt/share/x00960933/megamoe/` 下的 gen/run/一键脚本（第三方，已就绪） |

三套完全分开：PART1 只测基线网格 16 case，PART2 只测算子表 56 case，PART3 是独立 wheel 的 MegaMoE 测试链。归档目录各自独立（`perf-part1-*` / `perf-ep{8,32,64}-*` / `runs/*`）。

## 一、EP 档位与机器分配

| EP | 机器 | 每机卡数 | 总卡数 | 依据（清单） |
|----|------|------|------|------|
| EP8 | os1（单机） | 8 | 8 | D 组 EP8 算子表 |
| EP32 | os1~os4 | 8 × 4 | 32 | A1 DeepSeek + B1 K3 |
| EP64 | os1~os8 | 8 × 8 | 64 | A2 DeepSeek + B2 K3 |

## 二、性能 Case（共 56 个）

性能口径统一：`--route random --profile-op {dispatch,combine} --profile-warmup 5 --profile-iters 10 --repeat 2`
（与 test_shape.txt 基线一致；dispatch FP8 db=1 / combine BF16 db=2；weights0 即文档"不传 topk_weights"场景）

### EP8 Dispatch（D 组，8 个，单机 os1，dtype=fp8）
| # | Experts | Hidden | Tokens | TopK | 备注 |
|---|---------|--------|--------|------|------|
| 1 | 96 | 5120 | 64 | 6 | |
| 2 | 96 | 5120 | 256 | 6 | |
| 3 | 96 | 7168 | 64 | 6 | |
| 4 | 96 | 7168 | 256 | 6 | |
| 5 | 384 | 5120 | 4096 | 6 | |
| 6 | 384 | 5120 | 16384 | 6 | |
| 7 | 384 | 7168 | 4096 | 6 | |
| 8 | 384 | 7168 | 16384 | 6 | |

> D 组清单 dtype 未标注；按 test_shape 主口径取 FP8（可与基线 EP8 210-224 GB/s 对照）。

### EP32 Dispatch（A1+B1，24 个，4 机 os1~4，dtype=fp8）
| # | 组 | Hidden | TopK | Experts | Tokens |
|---|----|--------|------|---------|--------|
| 1-12 | DeepSeek | 7168 | 6 | 384 | 1 2 4 8 16 32 48 64 96 128 192 256 |
| 13-24 | K3 | 3584 | 16 | 896 | 同上 |

### EP32 Combine（A1+B1，24 个，4 机，dispatch-dtype=bf16）
与上同参数组、同 tokens 档。

### EP64 Dispatch（A2+B2，24 个，8 机 os1~8，dtype=fp8）
| # | 组 | Hidden | TopK | Experts | Tokens |
|---|----|--------|------|---------|--------|
| 1-12 | DeepSeek | 7168 | 6 | 384 | 1 2 4 8 16 32 48 64 96 128 192 256 |
| 13-24 | K3 | 3584 | 16 | 896 | 同上 |

### EP64 Combine（A2+B2，24 个，8 机，dispatch-dtype=bf16）
与上同参数组、同 tokens 档。

## 三、精度 Case（与性能一一对应）

精度口径：`--dtype both --route all --repeat 2`（dispatch，bf16+fp8 全路由）
combine 分两轮：`--dispatch-dtype bf16 --route all` + `--dispatch-dtype fp8 --with-dispatch --route all`

| EP | 脚本 | case 数 | 说明 |
|----|------|--------|------|
| EP8 | acc_dispatch_ep8（D 组 8 个）+ acc_combine_ep8 | 8+8 | 单机 os1 |
| EP32 | acc_dispatch_ep32 + acc_combine_ep32 | 24+24 | 4 机 os1~4 |
| EP64 | acc_dispatch_ep64 + acc_combine_ep64 | 24+24 | 8 机 os1~8 |

> 精度 = 搬运正确性（uint8 按位比较，dispatch）+ 浮点双黄金（combine double-golden，CPU FP32 + ordered Torch NPU）。

## 四、脚本清单

| 脚本 | 用途 | 运行环境 |
|------|------|---------|
| run_perf_dispatch_ep8_os1.sh | EP8 Dispatch 性能 | os1 单机 |
| run_perf_combine_ep8_os1.sh | EP8 Combine 性能 | os1 单机 |
| run_acc_dispatch_ep8_os1.sh | EP8 Dispatch 精度 | os1 单机 |
| run_acc_combine_ep8_os1.sh | EP8 Combine 精度 | os1 单机 |
| run_perf_dispatch_ep32_osN.sh | EP32 Dispatch 性能 | os1~os4 |
| run_perf_combine_ep32_osN.sh | EP32 Combine 性能 | os1~os4 |
| run_acc_dispatch_ep32_osN.sh | EP32 Dispatch 精度 | os1~os4 |
| run_acc_combine_ep32_osN.sh | EP32 Combine 精度 | os1~os4 |
| run_perf_dispatch_ep64_osN.sh | EP64 Dispatch 性能 | os1~os8 |
| run_perf_combine_ep64_osN.sh | EP64 Combine 性能 | os1~os8 |
| run_acc_dispatch_ep64_osN.sh | EP64 Dispatch 精度 | os1~os8 |
| run_acc_combine_ep64_osN.sh | EP64 Combine 精度 | os1~os8 |

## 五、部署与启动方式

### PART2（xlsx 算子表，已整理好）
- 模板在 `C:/Users/OY/WorkBuddy/A5/`，每机定制版由 `gen_dist.sh` 生成到 `A5/dist/`
- 定制版命名 `os{N}_run_*.sh`，已 scp 分发到各机 `/home/y00889330/`
- EP8 单机（os1 直接跑）：`bash /home/y00889330/run_perf_dispatch_ep8_os1.sh`
- EP32（4 台同时）：`bash /home/y00889330/os{N}_run_perf_dispatch_ep32.sh`（N=1..4）
- EP64（8 台同时）：`bash /home/y00889330/os{N}_run_perf_dispatch_ep64.sh`（N=1..8）
- 归档目录：`/home/y00889330/{perf,acc}-ep{8,32,64}-{dispatch,combine}/`

### PART1（test_shape 基线网格，本轮新写）
- 模板 `run_{perf1,acc1}_dispatch_ep{8,16,32,64}*.sh` 在 `A5/`，定制版由 `gen_dist_part1.sh` 生成到 `A5/dist-part1/`
- **EP8**（os1 单机）：`bash /home/y00889330/os1_run_perf1_dispatch_ep8.sh`（含 dispatch+combine 共 4 case）
- **EP16**（os1+os2 同时，master=os1）：`bash /home/y00889330/os{N}_run_perf1_dispatch_ep16.sh`（N=1..2）
- **EP32**（os1~os4 同时）：`bash /home/y00889330/os{N}_run_perf1_dispatch_ep32.sh`（N=1..4）
- **EP64**（os1~os8 同时）：`bash /home/y00889330/os{N}_run_perf1_dispatch_ep64.sh`（N=1..8）
- 精度版同名换 `acc1`：`bash /home/y00889330/os{N}_run_acc1_dispatch_ep{X}.sh`
- 多机带防失步握手（v5 waiting/go），master 自动调度 slave，无需人工干预
- 归档：`/home/y00889330/{perf,acc}-part1-ep{8,16,32,64}/`
- 环境：CANN b080（`/home/pkg/b080/ascend-toolkit/set_env.sh`），端口 19090/19091，SHMEM tcp master

## 六、PART3：MegaMoE（独立 wheel，2026-10-23 加入）

> 与 PART1/2 的 DeepEP 标准模式不同，MegaMoE 走 `deep_ep_experimental.megamoe.benchmark`（`megamoe.enabled=True` 的 wheel），**同一批机器、独立测试链**。

### 位置
- 目录：`/mnt/share/x00960933/megamoe/`（8 台共享盘）
- 一键性能：`run_perf_1click.sh`（环境检查→空闲检查→EP32→EP64→汇总）
- case 生成：`gen_cases_ep64.py` / `gen_cases_perf.py` → `cases/*.json` + `cases/configs_ep{8,16,32,64}/`
- 批量执行：`run_cases_multi.sh`（HOST_FILE/EP/MODE 环境变量驱动）
- 泛化口径：`deepep_mega.md`（E=每卡专家×EP，0<L≤128，S=0~8，B=1~65536，H=1024~8192 64倍数，I=256~4096 128倍数，K=1~min(32,E)；700 条：低延时 350 + 高吞吐 350）

### 性能 case（高吞吐，EP32/EP64）
| 项 | 值 |
|---|---|
| BS | 4096 |
| H / K / E / S / I | 7168 / 6 / 384 / 0 / 3072 |
| input | fp8 |
| warmup / iters | 10 / 20 |
| EP32（4 机 21-24）/ EP64（8 机 21-28） | 各 1 条，子集关系可直接对比扩展性 |

基线已跑：EP8=4453.477 us、EP16=4533.041 us；**EP32/64 待跑**（需 8 台全空闲）。

### 精度 case（accuracy，泛化 350/档）
- EP8 单机 os3：`runs/ep8-1023-203740/`（**2026-10-23 21:07 在跑 case_0247/350，勿打断**）
- EP16 双机：`runs/ep16-*`（已有 1022/1023 多轮，`EP16_精度汇总.md` 汇总）
- EP32/EP64：待 EP8 完成后按需跑
- 校验：`validate_ep64.py` / `validate_perf.py` / `summarize_ep16.py`

### 环境注意
- 8 台已统一装 MegaMoE wheel（`experimental=True megamoe.enabled=True`）；某节点被重置时：
  `pip install --force-reinstall --no-deps /mnt/share/x00960933/megamoe/wheels/ascend_deepep-*.whl`
- master 用 177.22.23.1（注意与 PART1/2 的 os1=177.22.21.1 不同）

