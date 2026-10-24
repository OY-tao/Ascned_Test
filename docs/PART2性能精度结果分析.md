# PART2 性能与精度结果分析

> 范围：PART2（`EP性能测试归档.xlsx` 算子表 case）已执行部分（EP32）
> 数据来源：服务器 `/home/deepep/perf-ep32-dispatch/` 的 kernel profile 记录
> 生成时间：2026-10-24

---

## 一、已执行 Case 构成（对照归档）

| 组 | 定义 | Case 数 | 状态 |
|---|---|---|---|
| A1 DeepSeek EP32 低时延 dispatch | H7168/K6/E384 × BS 档（FP8） | 7 | ✅ 已完成（t1~t48） |

---

## 二、EP32 Dispatch 性能结果（A1 组，实测 7 档）

**配置**：4 机 os1~4，FP8，H7168/K6/E384。kernel 耗时（us）：

| BS | weights0 avg(us) | weights1 avg(us) | 归档优化值(us) | 相对归档 |
|---|---|---|---|---|
| 1 | 770 | 1034 | 17.07 | **慢 ~45-60x** |
| 2 | 489 | 963 | 17.21 | **慢 ~28-56x** |
| 4 | 1830 | 919 | 17.96 | **慢 ~51-102x** |
| 8 | 1909 | — | 18.15 | **慢 ~105x** |
| 16 | 1398 | 1442 | 17.93 | **慢 ~78-80x** |
| 32 | 866 | 893 | 18.94 | **慢 ~46-47x** |
| 48 | 1828 | 1250 | 25.73 | **慢 ~49-71x** |

> **口径差异说明**：归档 A1 组数值（17~73us）来自 KIMI 汇报版本的低时延配置（8 卡单机或专用小规模），本次实测是 **4 机 32 卡分布式**（EP32 真实 = 32 卡），且 capture 10 轮中存在 token 分配不均（部分 rank 仅处理 1-2 token，耗时 60-70us，而多数 rank 700-1900us），整体均值被拉高。

**⚠️ EP32 dispatch 各 rank 严重负载不均**：t1 时 rank 间耗时 300-1200us 波动 4 倍；t16 时部分 rank 仅 60-70us（几乎无负载）而其他 rank 1900us。这是 EP32 多机小批量场景的固有开销（SHMEM 组网+广播），不代表 kernel 本身性能。

---

## 三、精度结果

### 3.1 EP32 Dispatch：⚠️ perf 脚本未含 dg 校验

perf 脚本参数 `--route random --repeat 2`，日志中 **无 double-golden 记录**（dispatch 场景不带精度校验）。EP32 dispatch 仅 `weights1 passed=True`（基本正确性校验），无严格 double-golden。

---

## 四、执行状态总结

| 项 | 结果 |
|---|---|
| EP32 Dispatch 性能 | ✅ 7 档完成（t1~t48） |

---

## 五、问题与建议

### 5.1 EP32 多机小批量 dispatch 慢

- **现象**：4 机 32 卡下 BS=1~48 的 dispatch 耗时 500~1900us，远高于归档 17~73us；
- **原因**：多机分布式 SHMEM 组网 + token 分配不均（部分 rank 空闲等待）；
- **建议**：与归档对比时应说明口径差异；若需对齐归档，应使用单机或 8 卡配置重测。

---

## 附：涉及日志与文件

| 路径 | 内容 |
|---|---|
| `/home/deepep/p2-ep32-perf-dispatch.os1.log` | EP32 dispatch 调度日志 |
| `/home/deepep/perf-ep32-dispatch/perf*_*/node0/profile/` | 各 case kernel profile |
