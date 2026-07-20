# MuJoCo Panda U 形工作台 RGB-D 环境

本项目提供一个统一、可配置、可复现的 Franka Panda 抓取与放置环境。在 U 形工作台、固定阻尼 DLS 和传统 RGB-D 感知之上，项目同时保留固定时序 B0，并新增由 RGB-D、本体状态和夹爪反馈驱动的事件控制 B1。

项目使用 MuJoCo 3.10.0 和只读的 `mujoco_menagerie` Panda submodule，不包含学习框架。

## U 形工作台

三块独立 box geom 的顶部均为 `z = 0.22 m`，U 形开口朝 `-X`：

| 区域 | body / geom | 中心 `(x, y, z)` m | 完整尺寸 `(x, y, z)` m |
| --- | --- | --- | --- |
| front | `u_table_front` / `u_table_front_geom` | `(0.55, 0.00, 0.20)` | `(0.60, 0.90, 0.04)` |
| left | `u_table_left` / `u_table_left_geom` | `(0.10, 0.57, 0.20)` | `(0.90, 0.24, 0.04)` |
| right | `u_table_right` / `u_table_right_geom` | `(0.10, -0.57, 0.20)` | `(0.90, 0.24, 0.04)` |

采样范围从已编译 geom 和 `edge_margin` 推导，不使用 IK 成功与否过滤。默认基座清空半径为 `0.20 m`，并额外计入物体半尺寸。

## 固定俯视 RGB-D 相机

唯一相机名为 `overhead_rgbd`，直接定义在 `worldbody`，不会跟随机器人或物体：

- 位置：`(0.70, 0.00, 2.00) m`
- 分辨率：`512 × 512`
- perspective vertical FOV：`60°`
- camera `+X` 世界轴：`(0.96949834, 0, -0.24509789)`
- camera `+Y` 世界轴：`(0, 1, 0)`
- camera `-Z` 观察方向：`(-0.24509789, 0, -0.96949834)`
- 相对垂直方向倾角：`14.1876°`

纯垂直视角下，默认 seed 42 的红色方块会被 Panda home 姿态严重遮挡，因此使用上述小倾角。实测三个桌面全部有效采样边界的最小图像边缘裕量约为 `96.96 px`。

## RGB、Depth 和相机几何

`OverheadRGBDCamera.capture()` 返回同一固定相机、同一仿真时刻和同一分辨率的 `RGBDFrame`：

- RGB：shape `(512, 512, 3)`，dtype `uint8`；
- Depth：shape `(512, 512)`，dtype `float32`；
- Depth 语义：以米为单位的相机光轴深度 `depth = -Z_camera`。

MuJoCo 3.10.0 `Renderer` 已经把 OpenGL reverse-Z buffer 线性化。Depth 不是欧氏射线距离，也不是 `[0,1]` buffer，因此代码不会二次套用 OpenGL 转换公式。无几何命中的背景可能是有限的 far-plane 深度；检测必须结合颜色 mask 和配置深度范围，不能只检查 `finite && > 0`。

内参由 vertical FOV 和分辨率计算：

```text
fx = fy = height / (2 tan(fovy / 2))
cx = (width - 1) / 2
cy = (height - 1) / 2
```

像素 `v` 向下，相机 `+Y` 指向图像上方，相机沿 `-Z` 观察。外参直接读取 `data.cam_xpos` 和 `data.cam_xmat`；后者是 camera-to-world 旋转矩阵。投影、反投影及 world/camera 变换位于 `perception/camera_geometry.py`。

## 传统 RGB-D 检测

当前场景只有一个固定形状红色立方体和一个绿色圆形目标。检测器只接收 `RGBDFrame`，不接收 object/site ID、采样坐标或 MuJoCo segmentation：

1. 使用可配置 RGB 通道阈值和颜色优势比生成 mask；
2. 过滤无效或范围外 depth；
3. 反投影到世界坐标并应用已知工作高度范围；
4. 选择最大的四连通区域；
5. 对 mask 内多个三维点取稳健中位数；
6. 红方块从可见表面沿世界 Z 减去已知半边长 `0.025 m`；
7. 绿色目标从可见顶面减去已知半厚度 `0.002 m`。

检测返回 mask、像素数、中心像素、三维位置、confidence 和结构化失败原因。当前算法没有使用神经网络或 OpenCV。

## privileged 与 perception

配置项 `[observation].source` 或 CLI `--observation-source` 选择外部任务状态：

- `privileged`：`PrivilegedStateProvider` 读取 MuJoCo 对象和目标真值，用于标签、调试及对照；
- `perception`：`RGBDPerceptionProvider` 只能通过 RGB-D、相机几何和检测器生成 `TaskStateEstimate`。

两种 provider 都使用稳定 ID `pick_object_0` 和 `place_target_0`；B0 在两种 source 下复用原 Fixed-DLS waypoint/状态机。B1 则使用下述独立事件状态机。在线控制器不读取 object body、target site 或 `current_episode`；只有独立 evaluator 读取真值计算误差。perception 模式的环境 observation 和 reset info 也不会暴露外部任务真值。

感知失败不会回退到 privileged provider，可能返回：

- `perception_object_not_found`
- `perception_target_not_found`
- `perception_invalid_depth`
- `perception_projection_error`
- `perception_low_confidence`

## B0 与 B1

`[controller].type` 或 CLI `--controller` 选择控制器：

| 控制器 | 定义 | 外部任务状态 | 阶段推进 |
| --- | --- | --- | --- |
| `fixed_dls_b0` | 原 Task 2 `FixedDLSPickPlaceController` | `privileged` 或 `perception` | 原固定时序；运行中会重新调用 provider |
| `sensor_event_b1` | `SensorEventPickPlaceController` | 只允许 `perception` | 位姿、速度、夹爪开口、双指接触、连续保持和超时事件 |

B0 的 `controllers/fixed_dls_controller.py`、固定阻尼公式、smoothstep 和原状态机保持不变。B1 只复用其纯 Fixed-DLS IK 数学：

```text
dq = J.T @ solve(J @ J.T + lambda^2 I, e)
```

B1 没有加入自适应阻尼或零空间项。smoothstep 仍生成关节参考，但参考时长结束不等于阶段成功；TCP 位置误差、姿态误差和最大关节速度进入阈值并连续保持 `arrival_hold_steps` 才到达。退出阈值使用 1.25 倍滞回，阶段达到 `motion_timeout` 时区分 `motion_stage_timeout` 和 `motion_not_settled`。

### 传感器职责与任务记忆

- RGB-D：初始物体/目标定位、预抓取物体重定位，以及撤离后的最终物体验证；下降、闭合、试抬、搬运和放置期间不要求继续看见物体。
- 本体反馈：7 个 arm joint 的位置/速度、TCP 位置/姿态和 actuator command 用于运动规划、到达与稳定判断。
- 夹爪编码器：`finger_joint1` / `finger_joint2` 的位置和速度用于开口、闭合趋势、释放与掉落推断。
- 二值接触代理：只报告左指接触、右指接触、双指接触及连续时长，用于抓取候选、确认和搬运监控。
- 状态机：决定当前阶段允许使用哪类数据。初始 `locked_target_position` 是静态目标的任务记忆，预抓取只更新物体位置，搬运和放置不会从 target site 重新读取目标。

初始和最终感知都对多帧有效三维位置取中位数，并记录有效帧数、最大离中位数距离、confidence 和总捕获延迟。预抓取默认不允许感知失败后使用初始位置；`allow_initial_object_fallback=false` 是保守默认。当前 camera-clear 预抓取和最终撤离偏移分别由 `pregrasp_observation_offset` / `final_observation_offset` 配置，避免夹爪从俯视相机遮挡红色物体。

### 夹爪开口与 simulated tactile proxy

实际 Menagerie Panda 模型中两指都是范围 `[0, 0.04] m` 的 slide joint，并由 identity joint equality 和 split tendon 同步。适配器构造时验证：

1. 两关节类型、范围和 equality；
2. 两指在世界系沿相反方向运动；
3. finger body 从闭合到张开的实际分离增量等于两关节行程之和。

因此本模型的可动开口定义为 `aperture = q_left + q_right`，名义范围 `[0, 0.08] m`，开口速度为两 joint velocity 之和。空夹关闭实测 qpos 和约为 `4.7e-8 m`，阈值不会假设求解器严格到零。

上游 finger collision geom 实际没有名称。`ContactSensor` 根据 `left_finger` / `right_finger` body 下 `contype != 0` 的 geom 动态建立集合，再把 `data.contact` 中的 finger/object geom pair 限缩为二值反馈；不会硬编码 geom ID。出现和消失都要求连续 `contact_debounce_steps`，同一仿真时间重复读取不会绕过 debounce。

该反馈是由 MuJoCo 接触对构建的 **simulated tactile proxy**，不代表已经接入真实硬件触觉传感器。它不向控制器暴露接触点、完整接触力、constraint force、物体位置或物体速度；当前也没有使用未经量纲验证的 actuator/contact force。

### B1 状态机契约

每次进入阶段都会记录入口仿真时间，退出或失败时写入 `stage_durations`。各处理器都有显式完成条件和有界 timeout：

| 阶段 | 使用的反馈与完成事件 | 主要失败 |
| --- | --- | --- |
| `scene_perception` | 多帧 RGB-D；物体和目标有效帧达到下限、spread 合格后锁定记忆 | `initial_perception_failed` |
| `move_to_pregrasp` | TCP 位姿、arm velocity；camera-clear 高位连续到达 | `motion_stage_timeout`, `motion_not_settled` |
| `pregrasp_reacquisition` | 只使用 RGB-D object detection；稳健更新下降点，目标记忆不变 | `pregrasp_reacquisition_failed`, `pregrasp_position_unstable` |
| `descend_to_grasp` | TCP 位姿、arm velocity；到达修正后的抓取位姿 | motion timeout / not settled |
| `close_gripper` | close command、开口、左右接触；出现去抖双侧接触 | `empty_gripper_closure`, `bilateral_contact_missing`, `grasp_candidate_failed` |
| `grasp_candidate_check` | 开口高于空夹阈值且双指接触连续保持 | candidate / bilateral / empty failure |
| `trial_lift` | TCP 小幅上移、开口和接触持续监控 | `trial_lift_failed` |
| `grasp_confirmation` | 试抬已到达、双指接触、开口无突然进一步闭合并连续保持 | `grasp_not_confirmed` |
| `transfer` | 锁定目标、TCP 事件到达；开口与双指接触掉落监控 | `grasp_lost_during_transfer`, motion failure |
| `descend_to_place` | 锁定目标和静态尺寸配置；到达放置位并继续监控夹持 | grasp-lost / motion failure |
| `release` | open command；开口超过 release 阈值并连续保持 | `release_failed` |
| `withdraw` | TCP 撤到 camera-clear 观察位并稳定 | motion timeout / not settled |
| `final_visual_verification` | 只需多帧 object detection；与锁定目标比较感知 XY 和桌面合理高度 | `final_object_not_found`, `final_visual_place_xy_error`, `final_visual_place_height_error` |
| `completed` | 最终视觉条件通过；`controller_reported_success=true` | 无 |

`发出 close command` 不等于抓取。`grasp_candidate` 需要非空开口、去抖双指接触和连续保持；之后先执行 `trial_lift_distance` 小幅试抬。只有试抬路标到达、双指接触和开口稳定再次连续保持，才进入 `grasp_confirmed`。搬运期不运行连续视觉伺服；只有“双侧接触持续丢失 + 开口朝空夹值进一步闭合”同时满足并超过阈值才进入 `grasp_lost`。单个 step 抖动会被 debounce 吸收；去抖后的短时 contact-loss 只记录事件，不会立即宣布掉落。

最终成功只来自撤离后的 RGB-D：最终物体中位数与 `locked_target_position` 比较，期望物体中心高度为 `locked_target_z - target_site_offset + object_half_size`。绿色目标此时可以不可见。MuJoCo 物体/目标真值仅在 `evaluation/perception_evaluator.py` 的终态 recorder 中计算 `privileged_ground_truth_success`、`false_positive` 和 `false_negative`，这些值不会回流控制器。

## 配置与复现

默认配置为 `configs/u_table.toml`，包括 pick/place/physics 的 fixed/random 模式、全局 seed、工作区安全边距、仿真参数、相机参数、感知阈值、原 Fixed-DLS 参数和完整 `[b1]` 阈值。B1 frame count、spread、camera-clear offset、到达/稳定、开口、contact hold、trial lift、掉落、release 和最终视觉容差均显式校验。

环境只维护一个 `numpy.random.Generator`。显式 `reset(seed=42)` 会重建 Generator，使抓取位置、目标位置、质量、摩擦和 reset 状态可复现。合法几何样本不会因为控制或视觉失败而重采样。

## 运行

```powershell
# B0 privileged 对照
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --controller fixed_dls_b0 --observation-source privileged --headless

# B0 RGB-D perception
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --controller fixed_dls_b0 --observation-source perception --headless

# B1 sensor/event baseline（B1 强制 perception）
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --controller sensor_event_b1 --observation-source perception --headless

# B1 Viewer
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --controller sensor_event_b1 --observation-source perception --viewer

# 捕获 RGB、metric depth、mask、NumPy 原始数组和相机元数据
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\capture_rgbd.py --config configs\u_table.toml --seed 42 --output-dir outputs\perception_debug

# 默认以 random pick/place 评测多个 seed
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\evaluate_perception.py --config configs\u_table.toml --seeds 42 43 44 45 46
```

`capture_rgbd.py` 输出 `rgb.png`、16-bit 毫米深度、8-bit 深度预览、两个 mask、原始 `.npy` 和 `metadata.json`。`outputs/` 已被 Git 忽略。

完整运行输出人类摘要和 JSON `EpisodeResult`。`to_dict()` / `to_json()` 保持兼容，`to_flat_dict()` 会展开 key error 和各阶段时长并把向量编码成标量字符串，可直接交给 `csv.DictWriter` 做批量统计。退出码为成功 `0`、可解释失败 `2`、未预期程序错误 `1`。

## 当前能力边界

当前只有单个固定尺寸红色立方体、单个静态绿色目标和单台固定俯视相机。B1 是传统事件控制基线，不包含连续视觉伺服、自适应 DLS、零空间优化、学习算法、神经网络视觉、形状随机化、多物体、多目标或动态目标；simulated tactile proxy 也不能替代真实硬件触觉标定。感知失败不会回退 privileged。

## 测试

```powershell
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe -m compileall environments controllers perception sensors evaluation scripts tests
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe -m unittest discover -s tests -v
```
