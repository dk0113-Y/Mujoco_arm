# CI-Baseline v1：隔离 Panda 固定增益笛卡尔阻抗自由空间基线

## 1. 目的与范围

CI-Baseline v1 在已验证的 JI direct-torque 基础设施上建立独立闭环：

```text
project-defined CI TCP pose/twist
→ fixed-gain 6D Cartesian impedance
→ J_world.T task-wrench mapping
→ verified gravity + Coriolis/centrifugal compensation
→ seven direct motor torques
→ isolated Panda dynamics
```

本版本只研究 headless 自由空间固定增益控制。没有接触控制、force feedback、
变阻抗、在线增益调节、零空间 torque、posture regulation、IK、MPC、学习、
抓放、RGB-D、ROS 2 或规划。它不接入 B0/B1、Development 或 Held-out。

允许的结论仅为：

> CI-Baseline 固定增益自由空间闭环已经建立并完成基础评测。

本实现不是 Franka 官方代码，不声明真机等价、真机精度或优于其他控制器。

## 2. 隔离与共享边界

CI 复用经过 JI 验证的：

- `models/panda_torque/scene_torque.xml` direct-torque plant；
- `PandaTorqueEnv` 七维物理 torque 接口和 joint safety；
- `MuJoCoDynamicsProvider`；
- 严格 JSON/CSV writer 和五产物输出约定。

唯一对 JI 隔离模型的受控扩展，是在 `panda_torque.xml` 的 `hand` 固定刚体
子树中新增无质量 site `ci_tcp`。它不改变 actuator、geometry、inertia、
collision、joint、tendon、equality 或 gripper。模型扩展由 TCP 刚体变换
测试覆盖；每次 CI 提交后完整复跑 JI。

未修改：

- 原 Menagerie Panda；
- B0/B1 模型、scene、controller、配置或注册；
- `PandaTorqueEnv`；
- JI controller、runner、配置、schema 或历史产物；
- `configs/baselines/`、`benchmark/`、`scripts/run_development.py`、
  `isaacsim/`。

## 3. 项目定义的 CI TCP site

`ci_tcp` 是本项目控制用 site，不称为真实 Franka K frame、`F_T_EE`、真机
flange 或其他未经证明的硬件 frame。

相对 `hand` body 的固定变换为：

```text
translation_hand_to_ci_tcp = [0, 0, 0.103] m
quaternion_hand_to_ci_tcp  = [1, 0, 0, 0]  (wxyz)
rotation_hand_to_ci_tcp    = I
```

自动测试在三个不同关节姿态下从 world pose 反算该相对变换，并验证其保持
不变。该平移与 B1 loader 动态创建的 `gripper_tcp` 数值定义一致，但两个
模型和运行链仍保持隔离。

## 4. 坐标、quaternion 和姿态误差

所有 CI 位置、旋转、速度、Jacobian、误差和 wrench 都用 MuJoCo world
frame 表达。

固定约定：

- rotation matrix：`R_world_tcp`；
- quaternion：scalar-first `[w,x,y,z]`；
- quaternion 每次转换都归一化；
- 日志采用 canonical hemisphere；`q` 与 `-q` 转换为相同 rotation；
- twist 和 wrench 顺序均为 `[linear xyz; angular xyz]`；
- Jacobian 与 orientation error 使用同一 world frame；
- 控制和指标不使用 Euler angle。

姿态误差定义为：

```text
R_error = R_target @ R_current.T
e_orientation_world = Log(R_error)
```

`Log` 返回 shortest-path axis-angle vector，模长为 geodesic angle，范围
`[0, pi]`。因此 world 正轴小角度目标产生同轴正误差。实现通过 normalized
relative quaternion 计算对数；接近 180 度时保持有限。quaternion sign
flip 不改变 rotation、orientation error 或 controller output。

## 5. TCP 运动学

`control_benchmarks/kinematics.py` 使用 MuJoCo 3.10 官方
`mj_jacSite`，返回：

- world TCP position `(3,)`；
- world rotation `(3,3)`；
- normalized canonical quaternion `(4,)`，顺序 `wxyz`；
- world TCP Jacobian `(6,7)`；
- linear/angular velocity 和 6D twist；
- singular values、numerical rank、minimum singular value、condition
  number；
- `J @ dq` 与 `mj_objectVelocity` 的最大逐元素差。

固定 Jacobian 约定：

```text
rows 0:3 = world linear xyz
rows 3:6 = world angular xyz
columns  = joint1 ... joint7
```

MuJoCo `mj_objectVelocity(..., flg_local=0)` 原生顺序是
`[angular; linear]`，provider 显式重排后再交叉核验。`mj_step` 在积分后
可能留下前一 forward stage 的派生缓存，provider 先执行 `mj_forward`，
确保 pose、Jacobian、site velocity 和当前 `q,dq` 对应同一状态。

自动验证包括：

- shape、joint 顺序和行序；
- world position 中心有限差分；
- `SO(3)` rotation-log 中心有限差分，不使用 Euler；
- 非零 `dq` 下 `J @ dq` 与 site velocity；
- `dq.T @ J.T @ wrench == (J @ dq).T @ wrench`；
- world X/Y/Z 平移和旋转的小目标短时闭环方向。

## 6. 固定增益控制律

`controllers/cartesian_impedance.py` 实现：

```text
e_p = p_target - p
e_R = Log(R_target @ R_current.T)

e_v = v_target - v
e_w = omega_target - omega

w_task =
    diag(K_translation, K_rotation) @ [e_p; e_R]
  + diag(D_translation, D_rotation) @ [e_v; e_w]

tau_task = J_world.T @ w_task
tau_raw  = tau_task + verified_dynamics_compensation
tau_rate = clamp(tau_raw, tau_previous ± torque_rate_limit * dt)
tau_cmd  = clamp(tau_rate, -torque_limit, +torque_limit)
```

正式固定增益为：

```text
K_translation = [120, 120, 120] N/m
K_rotation    = [8, 8, 8] N·m/rad
D_translation = [21.9089, 21.9089, 21.9089] N·s/m
D_rotation    = [5.6569, 5.6569, 5.6569] N·m·s/rad
```

这些是保守、显式、固定的 v1 参数，不是自动搜索或最优结果。结构对照
Apache-2.0 libfranka tag `0.21.2` simple Cartesian-impedance example，但
本实现采用适配 MuJoCo physical motor torque 的误差符号与补偿语义。

## 7. 动力学补偿

默认 mode 为 `gravity_coriolis`：

```text
tau_compensation = gravity + Coriolis/centrifugal
```

provider 在独立 scratch `MjData` 中计算同一 `q,dq` 的完整 bias，再以相同
`q`、零 `dq` 分离 gravity。passive joint damping 保留为 plant dynamics，
constraint force 不补偿。补偿只在 controller 的 `tau_raw` 中加入一次。

这与 FCI command 语义不同：Franka callback command 不含底层已补偿的
gravity/friction，而本 MuJoCo direct motor 是实际物理 torque，且模型
`body_gravcomp=0`，所以必须显式加入 gravity。controller 和环境都记录
rate/magnitude guard；环境第二道相同限制不会重复增加 compensation。

## 8. 可控性与安全门禁

配置固定：

```text
Jacobian numerical rank threshold = 6
rank tolerance                    = 1e-8
minimum singular value            = 0.05
maximum condition number          = 50
```

`J.T` 映射不求逆，也不使用 pseudoinverse 或 nullspace。rank、全部 singular
values、minimum singular value 和 condition number 每周期记录。正式 runner
在创建输出前对每个 episode 的 reset state 做可控性/contact preflight，并
对 201 个 task-trajectory sample 做 workspace 和 rotation validation；运行
中对实际 state 每周期继续 gate。

结构化终止原因包含：

```text
joint_position_limit
joint_velocity_limit
torque_saturation_sustained
torque_rate_limit_sustained
tcp_position_error_exceeded
tcp_orientation_error_exceeded
jacobian_rank_deficient
jacobian_condition_exceeded
invalid_orientation
unexpected_contact
non_finite_state
simulation_instability
timeout
```

其他固定门禁：

- actuator-compatible torque magnitude limit；
- `1000 N·m/s` torque-rate limit；
- MJCF joint range 内 `0.02 rad` soft margin；
- per-joint velocity limit；
- TCP position/orientation error 的 `0.25 s` sustained gate；
- `1000 rad/s²` simulation-instability threshold；
- finite checks；
- 任意 MuJoCo contact 都视为 free-space 的 `unexpected_contact`。

## 9. 正式自由空间轨迹

所有目标都以每个 episode reset 后的实际 TCP pose 相对生成，不调用 IK：

1. `cartesian_hold`：三个合法初始关节姿态，各保持 2 s；
2. `translation_axes`：world X/Y/Z 各 25 mm 平滑往返，固定姿态；
3. `orientation_axes`：world X/Y/Z 各 0.10 rad 平滑往返，固定位置；
4. `straight_line`：world `[0.04,0.02,0.03] m` minimum-jerk 直线；
5. `circle`：world XY 平面 25 mm 半径闭合圆。

axis trajectories 用 minimum-jerk window 平滑进入/退出；直线使用五次
minimum-jerk scaling；圆的 phase 从 0 以 minimum-jerk 前进到 `2*pi`，
所以位置闭合且起止 target velocity 为零。姿态轨迹用 world axis 的
`SO(3)` exponential map，不做 Euler 线性插值。

## 10. 日志、指标和产物

逐周期固定记录：

- episode identity、cycle、sim time；
- `q,dq`；
- TCP/target position、normalized `wxyz` quaternion、linear/angular
  velocity；
- position/orientation/linear/angular velocity error；
- task wrench、task torque；
- dynamics compensation、gravity、Coriolis/centrifugal、passive；
- raw/rate-limited/final torque 和 actuator force；
- Jacobian singular values/rank/minimum/condition、twist consistency error；
- torque/rate/joint/TCP/contact/finite masks 和 termination reason。

episode 指标包含 task-space position/orientation/velocity RMSE、最大值、
最终值、steady-state error、task force/moment、joint torque、limit counts、
Jacobian controllability、joint velocity、contact、finite/termination 和
sim/wall duration。orientation 指标使用 geodesic angle，不使用 Euler
per-axis RMSE 替代。

每次运行固定产生：

```text
run_manifest.json
episode_metrics.csv
timeseries.csv
summary.json
config_snapshot.toml
```

JSON 严格禁止 NaN/Infinity。CSV composite values 使用紧凑 JSON。非空输出
目录默认拒绝；`--overwrite` 只允许覆盖五个已知普通文件，未知文件或子目录
会导致拒绝。

## 11. 运行

先按项目验证流程设置并核验固定的 `$PY` 解释器，再运行：

```powershell
& $PY scripts/run_cartesian_impedance_benchmark.py `
    --config configs/control/ci_baseline_v1.toml `
    --experiment all `
    --output outputs/control/ci_baseline_v1
```

可选 experiment：

```text
cartesian_hold
translation_axes
orientation_axes
straight_line
circle
all
```

入口 headless、无 GUI、无网络依赖，也没有 Development/Held-out 参数。

## 12. 干净提交验证结果

本节在 feature commit 上完成正式 clean JI/CI 复跑后填写。最终性能结论只
引用 `git_dirty=false`、`git_commit` 与当时 HEAD 一致的原始产物。

## 13. 已知限制

- 500 Hz 不等于 Franka FCI 1 kHz；
- Menagerie Panda 是简化模型，不能外推真机性能；
- direct MuJoCo motor torque 与 FCI torque command 语义不同；
- 没有 link-side torque sensor、observer noise、communication latency、
  hardware reflex 或真实 friction；
- 无 nullspace posture regulation，冗余方向只受 plant passive damping；
- compensation 从零 torque 经过 rate limiter 建立，启动瞬态如实计入指标；
- `unexpected_contact` 使用仿真 solver truth 只做 safety/evaluation，不反馈
  给 controller；
- 本版本未研究接触、外力、payload/model mismatch 或增益变化。
