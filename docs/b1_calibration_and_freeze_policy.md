# B1 Calibration and Freeze Policy v1

## 1. 目的与边界

Calibration 只在 `calibration_v1.txt` 的全部 30 个 seed 上，用可诊断证据调整现有 B1-Vision 工程参数。它不自动搜索参数、不按成功率重写 TOML、不选择表现最好 seed 子集，也不宣布 B1 已冻结。正式冻结属于后续任务包 B；当前模板固定标记 `baseline_frozen=false`。

证据代码：P=RGB-D mask、像素/有效帧、confidence 和位置误差；M=阶段位姿/速度误差、到达保持、timeout 和碰撞；G=夹爪开口、双指接触、hold、trial lift、丢失/释放；V=最终帧、XY/高度误差与 controller/GT 对照。所有参数还必须通过 `environments.config.validate_config` 的跨字段约束。

## 2. 可校准参数（来自实际 dataclass）

以下 current value 由 `configs/baselines/b1_vision_calibration_template.toml` 和实际 `PerceptionConfig`、`ControllerConfig`、`B1Config` 字段核对。每项都允许 Calibration 调整，但冻结后均不得修改；“合法范围”之外还必须满足 config validator。

| 配置路径 | 当前值 | 单位 | 合法范围摘要 | 职责 | 必看证据 | 冻结后可改 |
|---|---:|---|---|---|---|---|
| perception.minimum_object_pixels | 30 | pixel | 正整数 | 物体 mask 最小面积 | P | 否 |
| perception.minimum_target_pixels | 80 | pixel | 正整数 | 目标 mask 最小面积 | P | 否 |
| perception.minimum_confidence | 0.50 | ratio | [0,1] | 检测接纳阈值 | P | 否 |
| perception.minimum_depth | 0.05 | m | >0 且小于 maximum | 有效深度下界 | P | 否 |
| perception.maximum_depth | 5.0 | m | 大于 minimum | 有效深度上界 | P | 否 |
| perception.object_min_rgb | [70,0,0] | RGB | 每项 [0,255] | 红色物体阈值 | P | 否 |
| perception.object_dominance_ratio | 1.70 | ratio | >1 | 红色优势比 | P | 否 |
| perception.target_min_rgb | [0,55,0] | RGB | 每项 [0,255] | 绿色目标阈值 | P | 否 |
| perception.target_dominance_ratio | 1.25 | ratio | >1 | 绿色优势比 | P | 否 |
| perception.object_world_z_range | [0.20,0.80] | m | lower < upper | 物体点云高度门限 | P | 否 |
| perception.target_world_z_range | [0.20,0.24] | m | lower < upper | 目标点云高度门限 | P | 否 |
| perception.object_surface_to_center | 0.025 | m | >=0 | 表面到物体中心修正 | P | 否 |
| perception.target_surface_to_center | 0.002 | m | >=0 | 表面到目标中心修正 | P | 否 |
| controller.ik_max_iterations | 500 | count | 正整数 | Fixed-DLS 迭代上限 | M | 否 |
| controller.ik_damping | 0.05 | unitless | >0 | 固定 DLS 阻尼 | M | 否 |
| controller.ik_step_gain | 0.70 | ratio | >0 | IK 更新增益 | M | 否 |
| controller.ik_max_joint_step | 0.08 | rad | >0 | 单次 IK 关节步长 | M | 否 |
| controller.ik_position_tolerance | 0.002 | m | >0 | IK 位置收敛 | M | 否 |
| controller.orientation_tolerance | 0.03 | rad | >0 | IK 姿态收敛 | M | 否 |
| controller.orientation_weight | 0.30 | ratio | >0 | 姿态任务权重 | M | 否 |
| controller.waypoint_height | 0.18 | m | >0 | 预抓/撤离高度 | M | 否 |
| controller.grasp_z_offset | 0.005 | m | >=0 | TCP 抓取 z 修正 | M,G | 否 |
| controller.lift_height | 0.20 | m | >0 | 搬运高度 | M,G | 否 |
| controller.approach_duration | 4.0 | s | >0 且小于 motion timeout | approach reference 时长 | M | 否 |
| controller.descent_duration | 3.0 | s | >0 且小于 motion timeout | descent reference 时长 | M | 否 |
| controller.lift_duration | 3.0 | s | >0 | lift reference 时长 | M,G | 否 |
| controller.transfer_duration | 4.0 | s | >0 且小于 motion timeout | transfer reference 时长 | M,G | 否 |
| controller.withdraw_duration | 3.0 | s | >0 且小于 motion timeout | withdraw reference 时长 | M,V | 否 |
| controller.gripper_open_control | 255.0 | command | 大于 close control | 打开命令 | G | 否 |
| controller.gripper_close_control | 0.0 | command | 小于 open control | 闭合命令 | G | 否 |
| b1.initial_perception_frames | 5 | frame | 正整数 | 初始多帧数 | P | 否 |
| b1.minimum_valid_perception_frames | 3 | frame | 1..initial frames | 初始最少有效帧 | P | 否 |
| b1.pregrasp_perception_frames | 3 | frame | 正整数 | 预抓重定位帧数 | P,M | 否 |
| b1.minimum_valid_pregrasp_frames | 2 | frame | 1..pregrasp frames | 预抓最少有效帧 | P,M | 否 |
| b1.pregrasp_observation_offset | [-0.08,0,0] | m | 有限 3-vector，norm<=0.20 | 预抓观察位偏置 | P,M | 否 |
| b1.maximum_position_spread | 0.02 | m | >0 | 多帧位置 spread | P | 否 |
| b1.maximum_pregrasp_correction | 0.08 | m | >0 | 预抓修正诊断阈值 | P,M | 否 |
| b1.allow_initial_object_fallback | false | bool | boolean | 重定位失败回退策略 | P,M | 否 |
| b1.arrival_position_tolerance | 0.015 | m | >0 | 事件到达位置阈值 | M | 否 |
| b1.arrival_orientation_tolerance | 0.05 | rad | >0 | 事件到达姿态阈值 | M | 否 |
| b1.settled_joint_velocity_threshold | 0.15 | rad/s | >0 | 稳定速度阈值 | M | 否 |
| b1.arrival_hold_steps | 15 | step | 正整数 | 到达/释放连续保持 | M,G | 否 |
| b1.motion_timeout | 7.0 | s | 大于最长 reference duration | 单运动阶段 timeout | M | 否 |
| b1.close_timeout | 2.5 | s | >0 | 闭合 timeout | G | 否 |
| b1.empty_gripper_aperture_threshold | 0.004 | m | >0，< minimum grasp | 空夹闭合阈值 | G | 否 |
| b1.minimum_grasp_aperture | 0.008 | m | > empty，< release，<=0.08 | 最小有效抓取开口 | G | 否 |
| b1.contact_debounce_steps | 3 | step | 正整数 | 接触去抖 | G | 否 |
| b1.bilateral_contact_hold_steps | 10 | step | 正整数 | 双指 candidate hold | G | 否 |
| b1.trial_lift_distance | 0.04 | m | >0 | 试抬距离 | G,M | 否 |
| b1.trial_lift_timeout | 4.0 | s | >0 | 试抬 timeout | G,M | 否 |
| b1.grasp_confirmation_hold_steps | 15 | step | 正整数 | 抓取确认 hold | G | 否 |
| b1.contact_loss_hold_steps | 25 | step | 正整数 | 接触丢失 hold | G | 否 |
| b1.aperture_drop_threshold | 0.003 | m | >0 且 <=0.08 | 进一步闭合/掉落阈值 | G | 否 |
| b1.release_aperture_threshold | 0.07 | m | > minimum grasp 且 <=0.08 | 释放开口阈值 | G | 否 |
| b1.release_timeout | 2.5 | s | >0 | 释放 timeout | G | 否 |
| b1.final_observation_offset | [-0.08,0,0] | m | 有限 3-vector，norm<=0.20 | 最终观察位偏置 | M,V | 否 |
| b1.final_verification_frames | 5 | frame | 正整数 | 最终验证帧数 | V | 否 |
| b1.final_minimum_valid_frames | 3 | frame | 1..final frames | 最终最少有效帧 | V | 否 |
| b1.final_place_xy_tolerance | 0.06 | m | >0 | controller 与独立真值 XY 容差 | V | 否 |
| b1.final_place_height_tolerance | 0.03 | m | >0 | controller 与独立真值高度容差 | V | 否 |

`controller.waypoint_tolerance`、`minimum_lift_height`、旧 fixed-time `gripper_duration/motion_hold_time` 等未被当前 SensorEvent B1 决策路径使用，不进入 v1 allowlist。若要重新启用或改变其职责，先升级 baseline/protocol，而不是在 Calibration 中暗调。

## 3. 禁止调整内容

Calibration 不得改变：U 形桌几何、Panda 模型、物体/目标尺寸、相机位姿/分辨率、random pick/place/physics 范围、任何 seed、成功/核心指标/碰撞/program error 规则、Oracle 权限与 B0/B1 成对公平性、状态机阶段、恢复机制、算法组件、自适应 DLS、零空间控制、视觉伺服或学习算法。也不得根据单个 seed 无限微调、根据 Development/Test 修改 B1，或缩小任务范围提高成功率。

若确认实现 bug 且修复会影响结果，立即停止 Calibration，说明影响，创建新 protocol 或 baseline 版本，重新生成受影响 split，并重新运行受影响实验。

## 4. 参数修改记录格式与停止条件

每轮必须保存首次/当前配置和结果，并记录：轮次、参数路径、旧值、新值、单位、修改理由、支持 seed/日志/图像证据、受影响失败类型、预期副作用、config SHA-256、code commit、protocol/split 版本。每轮只修改 allowlist 参数；不得自动挑选最佳 seed 子集。

建议停止条件：所有 30 个 Calibration seed 已至少完成一次记录；预先定义的工程问题已有足够证据；继续修改只针对单 seed 或会牺牲协议保护项；候选配置完成完整 Calibration 重跑和回归测试。停止不等于冻结，由用户审阅后决定。

## 5. B1-Vision v1 冻结步骤

1. 使用 Calibration v1 全部 30 seed，保存首次配置/结果。
2. 每轮只改 allowlist 参数，并保存上述变更记录。
3. 不从 Development/Held-out 获取调参信号。
4. 选定后生成 `configs/baselines/b1_vision_v1.toml`、calibration report、最终 config hash、code commit、protocol version 和 split version。
5. 运行 compile、全量 unit/regression、协议验证和完整 Calibration 复核。
6. 用户审查后决定是否创建明确版本标签；工具不自动 commit、tag、push 或宣布冻结。
7. 冻结后任何行为性改变产生 B1-Vision v2，并重跑相关实验。

允许的非行为性修改仅限文档、日志格式、不参与控制的诊断字段和拼写修正。行为性修改包括感知参数、状态机、路标、控制阈值、夹爪判定、成功条件、环境范围、相机和任何影响结果的 bug fix。

## 6. 版本与 Development/Test 限制

文档无行为错字修正可升 patch；增加可重算诊断可升 minor；任务、成功、指标、split、碰撞、随机范围或 Oracle 权限改变必须升 major。Development 只能用于后续方法开发和失败分析，不能修改冻结 B1；Held-out 只在方法冻结后使用。根据 Held-out 结果修改方法会使结果失效，必须使用新测试集或升级 protocol/split。
