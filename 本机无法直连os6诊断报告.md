# 本机无法直连 os6（177.22.26.1）诊断报告

> 版本：v1（2026-10-26 11:50）
> 现象：本机（Windows 客户端）ssh 直连 os6 超时/被 reset，需经 os1 跳板；本机连接 os1/2/7/8 均正常。

## 一、诊断结论

**根因：中间网络路径对"本机 → 177.22.26.1"的 SSH 流量在密钥交换（KEX）阶段主动中断（注入 RST/FIN），非本机配置问题、非 os6 防火墙问题、非 SSH 认证问题。**

本质证据链：
- TCP 三次握手成功、SSH 版本交换成功 → **但 KEX 阶段被 reset**（`expecting SSH2_MSG_KEX_ECDH_REPLY` 后 `Connection reset`）
- 同一来源 IP（`1.2.23.153`，本机经 NAT 后的来源）到 os1/os2/os7 **全部正常 ESTAB**，唯独到 os6 在 KEX 时被中断（os6 侧连接停留在 `FIN-WAIT-1`，sshd 从未进入认证阶段）
- os6 自身防火墙全开（INPUT ACCEPT）、无 hosts.deny/allow 限制、sshd 无 IP 白名单

## 二、实测证据表

| 测试项 | 结果 | 说明 |
|---|---|---|
| 本机 ping os6 (177.22.26.1) | ✅ 0% 丢失，42ms | 网络层通 |
| 本机 TCP 22 → os6 | ✅ 端口开放（可建连） | TCP 层通 |
| 本机 ssh -v 直连 os6 | ❌ `KEX_ECDH_REPLY` 等待后 reset | **KEX 阶段被中断** |
| 本机 ssh 直连 os1/2/7/8 | ✅ 全部正常（hostname 返回） | 对照组正常 |
| os1 ssh 直连 os6 | ✅ 正常（hostname 返回） | 非 os6 服务问题 |
| 本机 ProxyJump 经 os1 → os6 | ✅ 正常（返回 rack05os6） | **跳板路径可用** |
| os6 防火墙/访问控制 | 全开放、无限制 | 排除 os6 侧拦截 |
| os6 `ss -tn` 本机连接 | `1.2.23.153:54045/54046` 处 `FIN-WAIT-1` | **对端（中间设备）已发 FIN** |
| os6 sshd journal | 仅有 os1（177.22.21.1）记录，**无 1.2.23.153 认证记录** | 本机连接从未进入 sshd 认证 |
| os7/os2 sshd journal | `Connection from 1.2.23.153` 正常 ESTAB | 同源到其他节点正常 |
| traceroute 到 os6/os1 | 路径一致（第 3 跳直达，前两跳 ICMP 隐藏） | 路径无异常分叉 |

## 三、关键事实

1. **本机真实出网 IP**：`124.70.231.55`（公网）；**到 os6/os1 等显示来源 `1.2.23.153`**——中间存在一层 NAT/转发设备做源地址转换。
2. **同一来源 `1.2.23.153`**：到 os1（`Connection from 1.2.23.153` 正常）、os2（正常）、os7（ESTAB）均成功；**仅到 os6 被中断**。
3. **中断特征**：KEX（密钥交换）阶段注入 RST/FIN——`ss` 显示 `FIN-WAIT-1`（对端主动半关闭），客户端等 `KEX_ECDH_REPLY` 时收到 `Connection reset`。
4. **os6 自身无拦截**：iptables INPUT policy ACCEPT、无 hosts.deny、sshd 无 AllowUsers/DenyUsers/ListenAddress 限制。

## 四、推断

中间设备（位于 1.2.23.153 与 177.22.26.1 之间）对**该特定目的地址的 SSH 流量**存在状态跟踪/深度检测异常，或存在针对 `→177.22.26.1` 的访问控制策略。具体位置无法从客户端侧直接观测（traceroute 前两跳 ICMP 被隐藏），需联系网络管理员核查：
- 是否存在针对 `177.22.26.1` 的 ACL / 安全组规则（尤其 SSH 流量深度检测）
- NAT 设备（1.2.23.153）到 177.22.26.1 路径上的会话表状态同步问题（FIN-WAIT 残留）

## 五、绕过方案（已验证可用）

| 方案 | 状态 | 命令示例 |
|---|---|---|
| **ProxyJump 经 os1 跳板** | ✅ 已验证 | `ssh -J root@177.22.21.1 root@177.22.26.1` |
| **os1 上 base64 管道中转** | ✅ 已验证 | `cat file \| base64 \| ssh root@177.22.21.1 'base64 -d > file'`（写 os6 时再套一层） |
| os1 上 nc/socat 端口转发 | 可用（未测） | `ssh -L 2222:177.22.26.1:22 root@177.22.21.1` 后本地连 localhost:2222 |
| **修改 os6 白名单** | 需网络管理员 | 需联系运维确认中间 ACL |

## 六、当前影响

- 本机所有对 os6 的操作（远程命令、传文件）**必须经 os1 跳板**，无法直连
- 测试链路（y00889330/x00960933/b00943526 账号）中涉及 os6 的下发/回传已在脚本层适配（relay 跳板逻辑）
- PART1 EP64 三档失败与此网络问题**无直接关系**（os6 已通过跳板真实参与，8 台全失败根因是 HT 版 64 rank SHMEM 组网卡死，见 SHMEM 诊断报告）
