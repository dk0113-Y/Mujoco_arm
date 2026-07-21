# Evaluation Protocol v1

## 1. Problem Definition v1

面向随机化工作台抓放任务，在使用 RGB-D、本体状态和夹爪接触反馈的条件下，提高机械臂系统的一次任务成功率，减少碰撞和不可解释失败，控制完整任务节拍，并保证失败可诊断、实验可复现。

本协议评价完整的感知—规划—控制—接触—释放—验证闭环，而不是只评价 IK、视觉或单个状态机阶段。生产型主目标是任务可靠性、机器人—工作台碰撞安全、完整任务节拍和失败可诊断性。B0-Oracle 只量化理想外部状态相对 RGB-D 外部状态的损失，是诊断工具而非可部署方法；B1-Vision 是当前现实传统控制基线，后续正式改进优先使用 Vision 输入。解释性学术指标不会自动成为主要业务指标。

当前工作只在 MuJoCo 中构造生产型代理指标。它不构成工业安全认证，不代表真机部署、OEE 验证、sim-to-real 完成或真实触觉标定。

## 2. 正式任务配置与几何核对

正式配置是 `configs/protocols/evaluation_protocol_v1.toml`，可独立保存、计算 SHA-256 和归档；通用 `configs/u_table.toml` 保持 fixed 默认值。协议标识为 `evaluation_protocol/1.0.0`，metrics schema 为 `1.0.0`，split ID 为 `evaluation_protocol_v1`。正式任务强制 `pick.mode=random`、`place.mode=random`、`physics.mode=random`，episode timeout 为 35.0 s。

下表来自 `scenes/panda_u_table_scene.xml` 经 MuJoCo 编译后的 geom 中心/半尺寸，并应用 `edge_margin=0.055 m`：

| 区域 | 桌面原始范围 x/y (m) | 正式中心采样范围 x/y (m) | 顶面 z (m) |
|---|---|---|---:|
| front | x=[0.25,0.85], y=[-0.45,0.45] | x=[0.305,0.795], y=[-0.395,0.395] | 0.22 |
| left | x=[-0.35,0.55], y=[0.45,0.69] | x=[-0.295,0.495], y=[0.505,0.635] | 0.22 |
| right | x=[-0.35,0.55], y=[-0.69,-0.45] | x=[-0.295,0.495], y=[-0.635,-0.505] | 0.22 |

物体半尺寸为 0.025 m。Panda 基座清空判定使用 `base_clearance_radius + object_half_size = 0.225 m`。目标圆柱半径与 edge margin 均为 0.055 m，因此目标不会越出桌面；物体也保有至少 0.030 m 的额外边缘裕量。pick 与 place 独立选择区域，允许同区域和跨区域；二者三维中心距离用于分布报告，XY 距离必须至少为 0.18 m，所以初始物体与目标不会重叠。

物体质量均匀采样于 `[0.05,0.20] kg`。MuJoCo 三个摩擦分量分别采样于 sliding `[0.80,1.40]`、torsional `[0.005,0.02]` 和 rolling `[0.0005,0.002]`，所有上下界均为正。以上范围与 README 和实际 config dataclass 一致。

合法性检查逐 seed 调用一次环境 reset，只读 `current_episode` 与 reset 后的 MuJoCo 状态，不创建 Renderer，不导入或运行控制器，不执行 IK，也不读取控制结果。检查包括区域边界、基座清空、最小距离、z 高度、目标/物体重叠、质量/摩擦、非有限数、reset 碰撞和明显穿透。物体从 1 mm 间隙落到桌面后会保留 MuJoCo 接触求解器的亚毫米 overlap；检查器单独报告该值，超过 1 mm 才判为几何穿透错误。这只是检查容差，不改变环境或采样分布。

## 3. 结果语义

`controller_reported_success` 是共享状态机依据可用外部状态和传感反馈作出的判断。`privileged_ground_truth_success` 是独立 recorder 根据终态 MuJoCo 物体/目标真值和 B1 最终 XY/高度容差作出的判断。正式真实任务成功不得用 controller 结果替代。

`placement_success` 同时要求：

- 已执行释放；现有 B1 由终态阶段属于 `withdraw`、`final_visual_verification` 或 `completed` 证明；
- `privileged_ground_truth_success=true`，即最终物体中心 XY 误差不超过 0.06 m，高度误差不超过 0.03 m；
- `final_stage` 是声明的 B1 阶段；
- `simulation_time <= 35.0 s`。

placement 不要求零碰撞。因此最终放置成功但发生过机器人—工作台碰撞时，`placement_success=true`。

`safe_task_success` 进一步要求：

- `placement_success=true`；
- `collision_count=0`；
- 无 program error 或 unexpected exception；
- seed、method、fingerprint、任务元数据、终态、时间、碰撞和 controller/GT 结果等关键字段完整且类型合法；
- 在 timeout 内完成。

所以放置成功但发生碰撞时 `safe_task_success=false`。当前 B1 没有完整重新抓取/人工重试流程，`first_attempt_placement_success` 在 `full_regrasp_count=0` 时等于 placement；失败 seed 不自动重跑。

## 4. Program error 与不可解释失败

Program error 不伪装成普通任务失败，至少包括：未处理异常、provider/Renderer/环境等资源创建失败、清理异常、无效 pair、结果缺失、CSV/JSON 序列化失败、manifest/output finalization 失败和不可恢复 runner 错误。runner 即使继续处理后续 pair，最终也保持非零退出码。

下列任一条件计入不可解释失败：unexpected exception、program error、invalid pair、`unknown_failure`、真实失败但缺少 failure reason、关键字段缺失、非有限数、未声明终态、输出或清理错误。分母是全部请求 episode。`initial_perception_failed`、`motion_not_settled`、`bilateral_contact_missing`、`grasp_lost_during_transfer` 等已结构化失败不计入不可解释失败。

## 5. 五个核心生产型指标

1. `safe_task_success_rate = safe_task_success 数 / 有效且无 program error 的 episode 数`。关键字段缺失的有效 episode 留在分母中但不能成为安全成功，同时计入不可解释失败。
2. `first_attempt_placement_success_rate = 未完整重新抓取或人工重试而 placement 成功数 / 有效且无 program error 的 episode 数`。不得通过重跑失败 seed 提高该指标。
3. `collision_episode_rate = collision_count > 0 的有效且无 program error episode 数 / 有效且无 program error 的 episode 数`。这是 episode 比例；`collision_count` 只作诊断。
4. 成功任务周期时间主要报告安全成功 episode 的 MuJoCo `simulation_time` 中位数，并保留 count、mean、minimum、maximum。wall-clock latency 不是生产节拍主指标；仿真时间只是算法与任务流程代理，不等于真实工业节拍。
5. `unexplained_failure_rate = 不可解释失败 episode 数 / 全部请求 episode 数`。

空分母的 rate 输出 `null` 而非 0。NaN/Inf 不写入 JSON；它们被显式标为无效数值和不可解释失败。实现位于 `evaluation/production_metrics.py`，不依赖 pandas。

## 6. 解释性诊断指标

以下只用于定位，不是主要排名目标：failure reason 与最终阶段分布、各阶段仿真耗时、Oracle/Vision 的 both success/oracle only/vision only/both failed、controller false positive/negative、Vision 初始物体/目标误差、预抓取修正量、最终视觉物体误差、接触丢失事件、grasp candidate/confirmed、trial lift 通过、按 pick/place/区域组合/同跨区域分组、按质量/摩擦/抓放距离区间分组。v1 不新增 Jacobian 最小奇异值、manipulability 或 jerk；后续只在 Development 证据需要时增加非控制诊断。

## 7. Seed 划分与生成算法

正式文件为 Calibration 30、Development 60、Held-out Test 100，另有两个不属于三类正式集合的 smoke seed。文件顺序固定、seed 非负、组内唯一、三组互斥，且 smoke 也不得与正式集合重叠。

生成器 `splitmix64_stratified_coverage_v1` 使用固定生成 seed `20260721`，生成日期固定记录为 `2026-07-21`。它对候选整数 `[0,4096)` 用 SplitMix64 键确定排序，取 512 个候选并逐个做环境 reset。每个 split 先为九种 pick→place 区域组合各选一个样本，再平衡组合计数，并以质量、三个摩擦分量和抓放距离的四分箱新覆盖作确定性 tie-break。它不使用 controller、B0/B1、任务成败、failure reason、IK 可达性或人工难易判断。候选非法样本只能按预先声明的几何/数值规则排除；v1 实际候选非法数为 0。

`split_manifest.json` 记录生成器、候选范围、协议 SHA-256、每个 seed 文件 SHA-256、覆盖统计和自身规范 JSON SHA-256。相同输入必须重建完全相同的 seed 顺序和 manifest hash。

## 8. 三类集合的使用规则

Calibration 可反复运行、查看 Viewer/日志并调整 allowlist 中现有 B1 工程参数；它不用于选择 B2 最终方案或报告最终性能，也不得为结果修改协议。Development 用于 B0/B1 失败分析、选择 B2 方向、开发/调节/消融 B2；可反复运行，但不得反向修改冻结后的 B1，也不是最终泛化结果。Held-out Test 只在方法冻结后运行，不用于调参或选算法，也不应在开发期间反复查看。若查看结果后修改方法，原 held-out 结果失效，必须换未使用测试集或提升 protocol/split 版本。

## 9. 公平性、追踪与 Benchmark-0 接入

B0 与 B1 仍使用完全相同的 `SensorEventPickPlaceController`、Fixed-DLS、ControllerConfig、B1Config、状态机、路标、超时、夹爪和接触逻辑；唯一方法差异仍是外部状态 provider。协议补丁只在 episode 完成后增加记录/纯汇总，不向控制决策反馈。

协议运行在原 Benchmark 字段之外增加 protocol ID/version、metrics schema、split ID/name、placement/safe/first-attempt、collision episode、unexplained、区域组合、同/跨区、三维抓放距离、config SHA-256 和 code commit；原 seed、method、任务参数、controller/GT 结果、fingerprint、pair 字段继续保留。run manifest 归档协议/配置/seed hash、Git commit/dirty/submodule 和 runtime 版本。Calibration manifest 明确写 `calibration_run=true`、`baseline_frozen=false` 和 `automatic_parameter_search=false`。

## 10. 版本规则

- Patch（如 1.0.1）：错字或不影响行为、split、任务、成功条件和计算的说明修正。
- Minor（如 1.1.0）：增加不改变已有核心指标的诊断字段或方法兼容，原指标仍可重算。
- Major（如 2.0.0）：改变任务分布、随机范围、split、成功条件、核心指标、碰撞规则或 Oracle 权限边界。

任何影响正式比较公平性的改动都必须升级适当版本。改变环境/协议保护字段时停止 Calibration，创建新 protocol/baseline 版本，重新生成受影响 split 并重跑实验。
