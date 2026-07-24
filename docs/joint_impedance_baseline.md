# JI-Baseline v1：隔离 Panda 关节阻抗力矩控制基线

## 1. 目的与范围

JI-Baseline v1 建立一条不侵入 B0/B1 的独立闭环：

```text
q, dq
→ joint impedance
→ seven joint torques
→ MuJoCo direct motors
→ Panda dynamics
→ q, dq at the next 2 ms control period
```

它用于验证关节顺序、力矩方向、动力学补偿、控制周期、安全限制和
结构化评测，不是控制算法改进。本版本没有 Cartesian impedance、
Jacobian/nullspace、接触控制、抓取、RGB-D、在线学习、MPC 或强化学习。

## 2. 隔离模型和 actuator 语义

模型入口是 `models/panda_torque/scene_torque.xml`。机械模型
`panda_torque.xml` 派生自仓库固定的 MuJoCo Menagerie Panda commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`，沿用原 mesh 资产而不复制。
几何、惯量、碰撞、关节范围、tendon、equality 和夹爪 actuator 保持不变。
来源与 Apache-2.0 说明见 `models/panda_torque/PROVENANCE.md`。

前七个 actuator 的编译契约为：

```text
transmission = joint1 ... joint7, one-to-one and in order
gear = 1
gain = 1
bias = 0
activation dynamics = none
ctrlrange = forcerange = ±[87, 87, 87, 87, 12, 12, 12] N·m
```

因此 `data.ctrl[0:7]` 是七关节物理 motor torque，不是位置目标。
`actuator8` 仍是独立的 tendon position actuator，范围 `[0,255]`，不进入
七维 arm action。自动测试还验证正控制量相对零控制量产生同关节正向加速度、
forcerange 实际生效，以及原 Menagerie `panda.xml` SHA-256
`c5a92e6ff47e7282ea303ffe13530ffe150248a22bb4b349a9369881e52facf0`
没有变化。

## 3. 环境接口

独立环境是 `environments.panda_torque_env.PandaTorqueEnv`：

```python
observation = env.reset(qpos=None, qvel=None, seed=None)
observation, diagnostics = env.step(joint_torque)
```

`joint_torque` 必须是 shape `(7,)` 的有限浮点向量，单位 N·m。错误维度和
NaN/Inf 被拒绝；环境不接受位置 action，也不做未记录的归一化或缩放。

observation 包含：

- `joint_positions`：7 维，rad；
- `joint_velocities`：7 维，rad/s；
- `simulation_time`：s；
- `actuator_force`：7 维，N·m；
- `applied_generalized_force`：7 维，N·m；
- `joint_limit_margin`：到最近 joint range 边界的 7 维余量，rad；
- `control_cycle`：当前控制周期。

diagnostics 包含 commanded、rate-limited、clipped 和 actuator torque，
饱和、变化率、关节位置、速度、跟踪误差和数值稳定 mask，以及结构化
termination reason。reset 会清除上一周期 torque 和持续违规计数。同一
初始状态和 action 序列得到逐元素相同的状态序列。

## 4. 控制律

`controllers/joint_impedance.py` 实现七轴独立 spring-damper：

```text
tau_feedback =
    K * (q_target - q)
    + D * (dq_target - dq)

tau_raw =
    tau_feedback
    + verified_dynamics_compensation

tau_rate =
    clamp(tau_raw, tau_previous ± torque_rate_limit * dt)

tau_final =
    clamp(tau_rate, -torque_limit, +torque_limit)
```

配置中的实际刚度为
`[80,80,60,60,20,15,10] N·m/rad`，阻尼为
`[17.8885,17.8885,15.4919,15.4919,8.9443,7.7460,6.3249] N·m·s/rad`。
阻尼约为 `2*sqrt(K)`；这些值是低于固定 libfranka Panda 示例的保守
MuJoCo 基线，不是搜索得到的最优增益。

控制器检查所有输入和中间值的有限性，记录 raw/rate-limited/final torque
和 mask，并通过 `reset()` 清除上一周期 torque 状态。

## 5. 与 Franka 官方示例的对应关系

反馈结构对应 Apache-2.0 `libfranka` tag `0.21.2` 的
`examples/joint_impedance_control.cpp`：

```text
K(q_d-q) + D(dq_d-dq) + coriolis
```

固定来源和链接登记在 `docs/control_baseline_feasibility.md`。本实现没有
复制整个官方源文件，只复现方程结构并增加 MuJoCo 所需适配。

关键差异：

- Franka callback 为 1 kHz；本模型是 500 Hz；
- FCI torque command 不含机器人内部补偿的 gravity/friction；
- 直接 MuJoCo motor 是实际施加的物理 torque，模型 `body_gravcomp=0`；
- 因而默认 MuJoCo motor command 必须显式加入 gravity 和
  Coriolis/centrifugal；
- MuJoCo Menagerie 是简化模型，不等于真机惯量、摩擦、payload 和安全链；
- `actuator_force/qfrc_actuator` 是仿真真值，不是 Franka link-side torque
  sensor；
- 本实现没有 FCI reflex、collision thresholds、通信 jitter 或真机硬限位。

所以本模块是“官方风格的 MuJoCo 适配”，不是 Franka 官方代码，也不声明
与 Franka 内部控制器或真机性能等价。

## 6. 动力学补偿语义

独立 provider 位于 `control_benchmarks/dynamics.py`。依据审计和 MuJoCo
3.10 动力学定义：

```text
qfrc_bias = gravity + Coriolis/centrifugal
qfrc_passive = joint damping and other passive terms
qfrc_constraint = solver constraint generalized force
```

provider 不凭字段名直接返回 `qfrc_bias`。它在独立 scratch `MjData` 中：

1. 以当前 `q,dq` 做 forward dynamics，得到完整 bias；
2. 以相同 `q`、零速度再次求值，得到 gravity；
3. 用二者之差得到 Coriolis/centrifugal；
4. 分别记录 passive 和 constraint；
5. 根据显式 mode 选择 `none`、`gravity` 或
   `gravity_coriolis` compensation。

默认 mode 是 `gravity_coriolis`，即加入 gravity +
Coriolis/centrifugal。passive joint damping 保留为 plant dynamics，
constraint force 也不补偿；两者不会重复加入。provider 要求
`body_gravcomp=0`，否则直接拒绝初始化，以防 gravity 重复补偿。

## 7. 控制周期与参考轨迹

模型 timestep 是 `0.002 s`，配置 substeps 是 1：

```text
simulation step = 2 ms
control period = 2 ms
control frequency = 500 Hz
```

配置 loader 要求 `1/control_frequency == timestep*substeps`。所有 torque
rate、轨迹和持续违规窗口都由实际 `dt` 计算。

轨迹模块与控制器解耦，提供：

- 固定合法姿态 hold；
- 五次时间标度 minimum-jerk point-to-point，起止速度和加速度为零；
- 只驱动指定一个关节的低频 sine，使用 minimum-jerk 窗平滑进入/退出；
- 七关节小幅、错相、受限频率的平滑组合轨迹。

所有单位为 rad、rad/s、rad/s² 和 s。配置 loader 验证完整七维形状、有限值、
持续时间、速度/频率参数及轨迹幅值不会越过软关节范围。固定配置无随机轨迹
分支，生成结果可逐元素重复。

## 8. 安全门禁

环境和 runner 记录并执行：

- 每关节绝对 torque limit；
- 每关节 torque-rate limit，正式值 `1000 N·m/s`；
- 距 MJCF joint range `0.02 rad` 的软位置门禁；
- 每关节速度门禁；
- NaN/Inf 拒绝和每周期状态有限性检查；
- 每关节最大跟踪误差；
- 持续 `0.25 s` 的 torque saturation；
- 持续 `0.25 s` 的 torque-rate limiting；
- `1000 rad/s²` acceleration instability 门禁；
- 配置最大 episode duration。

termination reason 使用：

```text
completed
joint_position_limit
joint_velocity_limit
torque_saturation_sustained
torque_rate_limit_sustained
tracking_error_exceeded
non_finite_state
simulation_instability
timeout
```

安全门禁终止 episode，不会把违规后的数据伪装为正常完成。

## 9. 实验与指标

正式 `all` 包含 13 个 episode：

- 1 个 zero-torque gravity response；
- 1 个 verified dynamics-compensation-only hold；
- 3 个 joint-impedance hold 姿态；
- 7 个逐关节 sine tracking；
- 1 个 seven-joint smooth tracking。

逐控制周期记录 sim time、q/dq、目标、误差、feedback、gravity、
Coriolis/centrifugal、passive、raw/rate/final torque、actuator force 和所有
limit mask。episode 汇总包含每关节 position/velocity RMSE、最大误差、
torque peak/RMS、saturation/rate count 和比例、最大速度、末端/稳态误差、
overshoot、适用时 settling time、有限性、终止原因和 sim/wall duration。

输出固定为：

```text
run_manifest.json
episode_metrics.csv
timeseries.csv
summary.json
config_snapshot.toml
```

JSON 使用 `allow_nan=False` 等价的严格写入；CSV 采用固定字段顺序，数组和
mask 用紧凑 JSON 编码。非空输出目录默认拒绝；`--overwrite` 只会替换五个
已知产物，存在任何未知文件时拒绝操作。

## 10. 运行命令

必须使用项目解释器：

```powershell
$PY = "C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe"

& $PY scripts/run_joint_impedance_benchmark.py `
    --config configs/control/ji_baseline_v1.toml `
    --experiment all `
    --output outputs/control/ji_baseline_v1
```

可选 experiment 为 `zero_torque`、`compensation_hold`、
`impedance_hold`、`single_joint`、`multi_joint` 和 `all`。需要替换已知
结果时显式增加 `--overwrite`。入口 headless、无 GUI、无网络和无
Development/Held-out split 参数。

完整测试命令：

```powershell
& $PY -m unittest discover -s tests -v
```

## 11. 2026-07-24 实际验证结果

环境为 Python 3.12.10、MuJoCo 3.10.0。实现前原有 184 项测试通过；新增
38 项测试后，完整结果是：

```text
Ran 222 tests in 68.538s
OK
```

正式配置实际执行 13 个 episode、15,822 个 control period，总模拟时间
31.644 s。落盘复核结果：

- 12 个 episode 为 `completed`；
- zero-torque 在 0.144 s 以 `joint_velocity_limit` 结构化终止；
- 所有 episode 的 `finite_value_status=true`；
- torque saturation 共 0 次，所有 episode 比例均为 0；
- torque-rate limiting 共 320 个 joint-cycle incidence；各 episode 最大
  比例为 compensation hold 的 0.4952%；
- 全局最大绝对 torque 为 41.9275 N·m，低于对应 87 N·m MJCF 限制；
- 15,822 行中 final torque 与实际 actuator force 全部逐元素一致；
- JSON 严格解析通过，CSV 共有固定 26 个 timeseries 字段，没有 NaN 或
  Infinity。

### 11.1 Zero-torque diagnostic

全程 arm torque 为零。关节 4 最大速度达到 `2.0022 rad/s` 后触发门禁；
终止时该关节相对初始姿态移动约 `0.1639 rad`。这证明 motor 没有隐藏的
位置保持。该安全终止是诊断预期，不是性能失败基线。

### 11.2 Dynamics-compensation hold diagnostic

只施加核验的 gravity + Coriolis/centrifugal，并保留 torque-rate limit。
episode 完成 1.5 s，无 saturation、非有限值或安全终止。最大 position
RMSE 是关节 2 的 `0.08364 rad`；其最大误差为 `0.13860 rad`。这段漂移主要
来自 reset 后物理 gravity torque 从零按 rate limit 建立，而该诊断没有
反馈弹簧把已经发生的位移拉回；它不应被解读为 impedance hold 性能。

### 11.3 三个 joint-impedance hold

| case | 最大每关节 position RMSE (rad) | 最大绝对误差 (rad) | 最大 torque (N·m) | 结果 |
|---|---:|---:|---:|---|
| pose 1 | 0.005534 | 0.013177 | 31.713 | completed |
| pose 2 | 0.004265 | 0.010794 | 25.877 | completed |
| pose 3 | 0.009453 | 0.022132 | 41.927 | completed |

三组均无 saturation、NaN/Inf、明显发散或安全终止。

### 11.4 七个 single-joint tracking

下表只列被驱动关节自身的指标；耦合关节完整指标在
`episode_metrics.csv`。

| driven joint | position RMSE (rad) | maximum absolute error (rad) | 结果 |
|---:|---:|---:|---|
| 1 | 0.005616 | 0.015995 | completed |
| 2 | 0.009185 | 0.028551 | completed |
| 3 | 0.006817 | 0.019058 | completed |
| 4 | 0.005885 | 0.014597 | completed |
| 5 | 0.003283 | 0.004913 | completed |
| 6 | 0.004227 | 0.006588 | completed |
| 7 | 0.006391 | 0.010466 | completed |

七个关节全部运行，无 saturation 或安全终止。

### 11.5 Multi-joint tracking

七关节 position RMSE 为：

```text
[0.004252, 0.003700, 0.005891, 0.002775, 0.002209, 0.003145, 0.005609] rad
```

每关节最大绝对误差为：

```text
[0.012204, 0.008671, 0.015563, 0.008285, 0.003666, 0.004887, 0.009476] rad
```

episode 完成，无 saturation、NaN/Inf、明显发散或安全终止。

## 12. 实际产物

实际结果位于：

```text
outputs/control/ji_baseline_v1/run_manifest.json
outputs/control/ji_baseline_v1/episode_metrics.csv
outputs/control/ji_baseline_v1/timeseries.csv
outputs/control/ji_baseline_v1/summary.json
outputs/control/ji_baseline_v1/config_snapshot.toml
```

`timeseries.csv` 约 36.8 MB，`outputs/` 由 `.gitignore` 排除，不提交到 Git。
性能结论来自这些实际文件的二次解析，而不是仅来自终端输出。

## 13. 已知限制

- 500 Hz 不是 Franka FCI 的 1 kHz；
- direct motor 物理 torque 与 FCI “不含内部 gravity/friction”的 command
  语义不同；
- torque-rate limiter 对完整 MuJoCo motor torque 生效，reset 后 gravity
  torque 的建立会产生 compensation-only transient；
- passive joint damping 未补偿，真实 Panda friction 也未建模；
- 没有 torque sensor、observer noise、latency、payload model 或硬件 reflex；
- Menagerie Panda 是简化模型，结果不能外推到真机性能；
- 没有接触、外力扰动、TCP trajectory 或 Cartesian impedance；
- settling time 只对固定 hold 定义；周期 tracking 使用 `null`；
- 当前结果只证明隔离的 JI 链路和基础安全/记录语义。

## 14. 进入 CI-Baseline 前的门禁

JI v1 的 direct torque、方向、限制、动力学分解、三姿态保持、七关节单独
跟踪、多关节跟踪、有限性、输出和回归测试门禁已经通过，因此可以开始独立
CI free-space 基础设施工作。但在声称 CI-Baseline 成立前仍必须单独完成：

1. 固定并测试 TCP site 与 Franka EE/K frame 的刚体变换；
2. 验证世界坐标 Jacobian、twist 行顺序和 `J^T` torque 符号；
3. 固定 quaternion hemisphere 和 orientation-error convention；
4. 建立 singularity/rank/condition safety；
5. 先复现官方 simple Cartesian impedance，不增加 nullspace；
6. 记录 compensation-only reset transient 的 CI 初始化策略；
7. 重新运行完整 JI gate，且不得接入 B0/B1、Development 或 Held-out。

这表示“允许进入下一阶段的隔离实现”，不表示 CI、接触控制或算法改进已经
实现或验证。
