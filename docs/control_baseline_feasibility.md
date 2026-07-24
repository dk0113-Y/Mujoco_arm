# Franka Panda 阻抗控制基线选型与可行性核验

审计日期：2026-07-24  
审计仓库：`dk0113-Y/Mujoco_arm`  
审计范围：只读核验、官方实现对照、接口映射和实验方案设计  
最终等级：`FEASIBLE_WITH_ISOLATED_MODEL_VARIANT`

## 1. 执行摘要

### 结论

当前模型**不能**把 `data.ctrl[0:7]` 解释为七个机械臂关节力矩。七个臂执行器虽然写成 `<general>`，但其编译后语义是高刚度位置伺服：

```text
p_i = Kp_i * ctrl_i - Kp_i * q_i - Kv_i * dq_i
    = Kp_i * (ctrl_i - q_i) - Kv_i * dq_i
```

随后执行器力分别限制在 `±[87, 87, 87, 87, 12, 12, 12] N·m`。证据是 Menagerie `panda.xml:9-12,264-277`、MuJoCo 3.10 actuator 仿射力公式，以及本次编译模型探针。

建立正式力矩控制基线是可行的，但必须新增与 B1 完全隔离的 torque-control 模型/环境变体。该变体应保留当前几何、惯量、碰撞、关节、场景和夹爪位置执行器，仅把前七个执行器改为 `gear=1`、固定增益 1、零 bias 的直接驱动，并保留明确的力矩范围。当前 B1 继续加载原始 Menagerie `panda.xml`，因此无需改变冻结行为。

分层判断：

| 项目 | 结论 | 进入条件 |
|---|---|---|
| JI-Baseline | 可行 | 先建立独立力矩模型；验证关节顺序、方向、重力/科氏补偿、饱和与变化率限制 |
| CI-Baseline | 条件可行 | 通过全部 JI 门禁后，再验证 TCP 定义、世界坐标 Jacobian、姿态误差和奇异性 |
| 接触实验 | 可行但需诊断接口 | MuJoCo 可给出 solver contact wrench、接触距离和 body wrench；当前项目适配器只暴露布尔接触 |
| 后续 ROS 2 映射 | 接口层面可行 | 当前核对的 `franka_ros2 v2.5.1` 面向 FR3，不是 Panda 的即插即用等价物 |
| 进入实现阶段 | **有条件允许** | 只允许先实现隔离模型、动力学适配器、保护层和 JI；不得宣称 CI 或接触控制已跑通 |

当前最大风险是把 Franka FCI 的“**不含重力和摩擦的 torque command**”直接照搬成 MuJoCo motor 的“**实际施加力矩**”。Franka 官方示例只显式加入 coriolis，是因为机器人底层另做重力/摩擦补偿；本模型 `body_gravcomp=0`，直接 motor 变体必须显式处理 gravity，且不得重复补偿。

本次未实现控制器、未修改任何 Python/MJCF/XML/配置/测试、未运行正式 Development、未读取或运行 Held-out Test。

## 2. 仓库和环境核验

### 2.1 命令与结果

在审计开始、工作区仍未新增本报告时执行：

```powershell
git status --short --branch
git rev-parse HEAD
git submodule status
git log -1 --oneline
python --version
python -c "import mujoco; print(mujoco.__version__)"
python -m unittest discover -s tests -v
```

实际结果：

| 项目 | 实际值 | 与附件预期的差异 |
|---|---|---|
| 分支 | `main`，跟踪 `origin/main` | 一致 |
| 初始工作区 | clean | 一致 |
| HEAD | `2c13723fbff14fa7b33e249cd7a4950376c20282`，`Merge feat/isaacsim-readiness into main` | 不同于预期 `8336bcea...` |
| 预期 HEAD 关系 | `8336bcea...` 是实际 HEAD 的第一父提交；新增差异仅为 `isaacsim/` 四个文件 | 控制链路未因该 merge 改变 |
| 子模块 | `71f066ad0be9cd271f7ed58c030243ef157af9f4 models/mujoco_menagerie` | 一致；子模块 detached、clean |
| PATH 中 `python` | `F:\Python312\python.exe`，Python 3.12.8 | 不同；且 `import mujoco` 失败 |
| `py -3.12` 对应解释器 | `C:\Python312\python.exe`，Python 3.12.10 | 版本匹配，但初始同样未安装 MuJoCo |
| 审计临时环境 | Python 3.12.10、MuJoCo 3.10.0、NumPy 2.5.1 | Python/MuJoCo 匹配 |
| 完整测试 | `Ran 184 tests in 65.929s`，`OK` | 与预期一致 |

为不修改仓库和全局 Python，本次在系统临时目录用 `C:\Python312\python.exe -m venv` 建立一次性环境，再执行：

```powershell
python -m pip install -r requirements.txt
python --version
python -c "import mujoco; print(mujoco.__version__)"
python -m unittest discover -s tests -v
```

仓库 `requirements.txt:1-2` 指定 `mujoco==3.10.0` 和未固定版本的 NumPy。可复现性注意点是：默认 PATH 环境不能直接运行项目，且 `requirements-lock.txt` 没有锁定 MuJoCo、其 NumPy 版本又与临时安装结果不同。

### 2.2 测试结论

184 项测试全部通过，没有异常可报告。测试覆盖了模型加载、headless smoke、B0 回归、B1 freeze/truth isolation、控制状态机、接触代理、夹爪反馈、Development 工具和协议隔离等。测试通过只能证明现有 B0/B1 行为，没有证明力矩控制或阻抗控制已存在。

### 2.3 冻结文件核验

```text
configs/baselines/b1_vision_v1.toml
SHA-256 6808c142ae8805695fc43d5e4743a9529cdbea15008810456184e40e1c4b7ea9

configs/baselines/b1_vision_v1_manifest.json
SHA-256 cc3e0f48a2821f2746cac250e7506ba877aa51ef53999e6fdc505580fc0c98d6
```

两文件相对 HEAD 无差异。`scripts/run_development.py:22-47,103-196,238-265` 固定了协议、B1 配置、Development split、freeze manifest、方法顺序和冻结 commit，并在调用 runner 时设置 `baseline_frozen=True`、`development_run=True`。

## 3. 当前控制数据流

### 3.1 实际入口到 actuator

```text
scripts/run_development.py:238-265
  -> benchmark.run_benchmark
benchmark/runner.py:565-570,619,700-714
  -> benchmark.methods.resolve_methods
benchmark/methods.py:46-72
  -> SensorEventPickPlaceController（B0/B1 共用）
benchmark/runner.py:220-295
  -> PandaUTableEnv(config)
  -> controller.run_episode(...)
controllers/sensor_event_controller.py:499-568
  -> solve_pose_ik(...)
  -> 7 维关节位置 target
  -> smoothstep 插值写入 data.ctrl[arm_actuator_ids]
controllers/sensor_event_controller.py:402-408
  -> env.step(env.data.ctrl.copy())
environments/panda_u_table_env.py:422-440
  -> 8 维检查、逐 actuator ctrlrange 裁剪
  -> data.ctrl[:]
  -> mj_step
models/.../franka_emika_panda/panda.xml:264-277
  -> 7 个 joint position servo + 1 个 tendon position servo
```

`benchmark/methods.py:57-72` 注册 `b0_oracle` 和 `b1_vision`，两者都使用 `SensorEventPickPlaceController`；区别是外部状态 provider，而不是控制器或 actuator。

### 3.2 `step(action)` 和 action 语义

环境类是 `environments/panda_u_table_env.py:100` 的 `PandaUTableEnv`；实际方法名为 `step(control)`，实现位于 `:422-462`。

环境输入契约：

- 形状必须为 `(model.nu,)`，当前 `model.nu=8`（`:427-431`）。
- 必须全部有限（`:432-433`）。
- 对每个分量按 `model.actuator_ctrlrange` 裁剪，之后原样写入 `data.ctrl`（`:434-438`）。
- 不做归一化、比例缩放、单位变换或偏置。
- `frame_skip` 次 `mj_step`（`:439-441`）。

这里的 8 维量不是统一物理单位：

| 索引 | actuator | Python 侧含义 | 最终 actuator 含义 |
|---|---|---|---|
| `0:7` | `actuator1..7` | IK 得到并平滑插值的关节位置目标，rad | 高刚度关节位置伺服参考 |
| `7` | `actuator8` | 夹爪控制 `0..255` | tendon 位置伺服参考；0 约为闭合，255 约为 0.04 m tendon length |

`controllers/fixed_dls_controller.py:39-45` 还有一个高层 `Action` 数据类，但它是状态机动作描述，不是直接传给环境的向量。B1 实际由 `solve_pose_ik` 产生关节位置，见 `controllers/fixed_dls_controller.py:76-155`；B1 在 `controllers/sensor_event_controller.py:499-539` 复用该 IK，先按 actuator ctrl range 裁剪，然后在 `:562-568` 以 `smoothstep` 从当前 control 插值到目标。

夹爪独立于七臂关节：

- 臂 actuator id 来自 `actuator1..7`，夹爪 id 单独取 `actuator8`：`environments/panda_u_table_env.py:27-31,152-161`。
- B1 关闭和打开夹爪只写 `gripper_actuator_id`：`controllers/sensor_event_controller.py:831-845,1030-1038`。
- 冻结值为 open `255.0`、close `0.0`：`configs/baselines/b1_vision_v1.toml:81-89`。

## 4. Panda MJCF 与 actuator 审计

### 4.1 实际模型装载与组合

当前没有一个静态的项目顶层 MJCF include 链。实际编译路径是：

1. `environments/panda_u_table_env.py:15-24` 指向子模块 `franka_emika_panda/panda.xml` 和项目 `scenes/panda_u_table_scene.xml`。
2. `:45-80` 用 `ElementTree` 读取两棵 XML，把项目 scene 的 `worldbody` 和 `visual` 元素追加到 Panda 根节点。
3. `:81-94` 在 `hand` 下新增 `gripper_tcp` site，位置 `[0,0,0.103]`。
4. `:95-97` 通过 `MjModel.from_xml_string` 编译内存中的合并模型。

因此 Menagerie 自带 `scene.xml` 没有被当前环境加载，项目 scene 也没有 `<include>`。当前几何、惯量、碰撞来自子模块 `panda.xml`；桌面、物体、目标和相机来自 `scenes/panda_u_table_scene.xml:1-41`。

Menagerie 子模块固定在 commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`。其 README 明确称该 Panda MJCF 是简化描述、源自公开 URDF，并说明“Added position-controlled actuators for the arm”，见 `models/mujoco_menagerie/franka_emika_panda/README.md:10-16,22-46`；许可证是 Apache-2.0（`:60-62`）。

### 4.2 编译模型尺寸和全局选项

MuJoCo 3.10 只读探针结果：

```text
nq=16, nv=15, nu=8, na=0, njnt=10, nbody=16, nsite=2, nsensor=0
timestep=0.002 s
integrator=mjINT_IMPLICITFAST
gravity=[0, 0, -9.81] m/s^2
```

`nq/nv` 除九个机器人关节外还包含场景物体的 free joint。模型没有 MJCF sensor；项目当前 contact adapter 直接读取接触列表。

### 4.3 关节属性

共同默认值来自 `panda.xml:6-13`。编译探针确认：

| joint | 类型 | range | damping | frictionloss | armature | stiffness | body gravcomp |
|---|---|---:|---:|---:|---:|---:|---:|
| joint1 | hinge | `[-2.8973, 2.8973]` | 1 | 0 | 0.1 | 0 | 0 |
| joint2 | hinge | `[-1.7628, 1.7628]` | 1 | 0 | 0.1 | 0 | 0 |
| joint3 | hinge | `[-2.8973, 2.8973]` | 1 | 0 | 0.1 | 0 | 0 |
| joint4 | hinge | `[-3.0718, -0.0698]` | 1 | 0 | 0.1 | 0 | 0 |
| joint5 | hinge | `[-2.8973, 2.8973]` | 1 | 0 | 0.1 | 0 | 0 |
| joint6 | hinge | `[-0.0175, 3.7525]` | 1 | 0 | 0.1 | 0 | 0 |
| joint7 | hinge | `[-2.8973, 2.8973]` | 1 | 0 | 0.1 | 0 | 0 |
| finger_joint1 | slide | `[0, 0.04]` | 1 | 0 | 0.1 | 0 | 0 |
| finger_joint2 | slide | `[0, 0.04]` | 1 | 0 | 0.1 | 0 | 0 |

具体 joint 元素位于 `panda.xml:135-144,147-159,165-178,197-220,230-232`。所有关节启用位置 range 限制。`jnt_actfrclimited=0`，即没有独立的“一个关节上所有 actuator 合力”范围；下节列出的限制是每个 actuator 自身的 `forcerange`。
MJCF 没有为这些 hinge joint 定义速度、加速度或 jerk 限制，也没有独立 joint-level torque range；这些都需要新控制保护层补充。

### 4.4 臂 actuator

XML 元素见 `panda.xml:264-274`，编译后全部为：

```text
type/general shortcut: <general>
transmission: mjTRN_JOINT
gear: [1,0,0,0,0,0]
dyntype: mjDYN_NONE
gaintype: mjGAIN_FIXED
biastype: mjBIAS_AFFINE
ctrllimited: true
forcelimited: true
```

| actuator | joint | ctrlrange（rad） | gainprm[0] | biasprm `[b0,b1,b2]` | forcerange（N·m） |
|---|---|---:|---:|---:|---:|
| actuator1 | joint1 | `[-2.8973,2.8973]` | 4500 | `[0,-4500,-450]` | `[-87,87]` |
| actuator2 | joint2 | `[-1.7628,1.7628]` | 4500 | `[0,-4500,-450]` | `[-87,87]` |
| actuator3 | joint3 | `[-2.8973,2.8973]` | 3500 | `[0,-3500,-350]` | `[-87,87]` |
| actuator4 | joint4 | `[-3.0718,-0.0698]` | 3500 | `[0,-3500,-350]` | `[-87,87]` |
| actuator5 | joint5 | `[-2.8973,2.8973]` | 2000 | `[0,-2000,-200]` | `[-12,12]` |
| actuator6 | joint6 | `[-0.0175,3.7525]` | 2000 | `[0,-2000,-200]` | `[-12,12]` |
| actuator7 | joint7 | `[-2.8973,2.8973]` | 2000 | `[0,-2000,-200]` | `[-12,12]` |

MuJoCo 3.10 官方 actuator 公式是

```text
scalar_force = gain_term * ctrl + bias_term
bias_term = b0 + b1 * length + b2 * velocity
```

对 gear=1 的 hinge joint，length/velocity 对应关节位置/速度，所以本表严格得到位置伺服公式。官方依据：

- [MuJoCo 3.10 computation / force generation](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/doc/computation/index.rst#L369-L394)
- [MuJoCo 3.10 XML `general`](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/doc/XMLreference.rst#L5572-L5602)
- `<position>` shortcut恰好编译成 `gainprm=kp`、`biasprm=0 -kp -kv`：[XML reference](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/doc/XMLreference.rst#L5713-L5727)。

结论：当前 `data.ctrl[0:7]` 绝不是力矩。

### 4.5 夹爪 actuator

`panda.xml:253-262` 用固定 tendon `split` 平分两指关节，并用 equality 保持两指相等；`actuator8` 位于 `:275-277`：

```text
transmission = tendon "split"
ctrlrange = [0,255]
forcerange = [-100,100]
gainprm = [0.01568627451,0,0]
biasprm = [0,-100,-10]
```

其 scalar force 是：

```text
p = 0.01568627451*u - 100*l - 10*ldot
```

静态目标 tendon length 为 `0.0001568627451*u`，所以 `u=255` 对应约 0.04 m，`u=0` 对应闭合。Python 环境没有做这一步映射；映射发生在 actuator 内部。`sensors/gripper_feedback.py:30-38,59-69,158-191` 另以两指位置之和给出 `[0,0.08] m` jaw aperture。

### 4.6 torque-control 变体判断

最小变化是仅替换 `actuator1..7`：

```xml
<motor name="actuatorN" joint="jointN" gear="1"
       ctrlrange="..." forcerange="..."/>
```

MuJoCo 3.10 `<motor>` 是 direct-drive shortcut，编译成 `gainprm=1`、无 bias、无 activation；官方依据见 [XML reference](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/doc/XMLreference.rst#L5633-L5652)。在 `gear=1`、单 actuator/单 hinge joint 下，`ctrl_i` 才能与 joint generalized torque 一一对应。夹爪 `actuator8` 可原样保留。

建议采用**独立的、项目自有的 torque MJCF 复制变体**，并用自动测试约束其非 actuator 部分与固定 Menagerie commit 保持一致：

- 单纯 `<include panda.xml>` 无法删除已包含的七个位置 actuator，继续追加 motor 会得到重复 actuation。
- 直接在编译后的 `MjModel.actuator_gainprm/biasprm/ctrlrange` 上运行时修改，容易遗漏 limit/type 派生状态，且实验 manifest 难以审计。
- 复制变体会有上游漂移风险，但可以通过记录来源 commit、只改 actuator block、编译属性 parity test 和模型 hash 管理。
- 一个同样可接受的后续实现是“独立 loader 在编译前对 XML 做确定性替换”；它必须有独立类/配置和模型 hash，不能复用或改写 B1 loader。它不同于修改已编译模型。

新变体能继续使用同一批 mesh asset、几何、惯量、碰撞和项目场景。只要 B1 的 `PANDA_XML_PATH`、配置和入口不变，新模型不会影响冻结结果。

## 5. 控制周期和仿真步长

编译模型 `model.opt.timestep=0.002 s`；来源 XML 只指定 `integrator="implicitfast"`（`panda.xml:2-4`），timestep 使用 MuJoCo 默认值。B1 `frame_skip=1`（`configs/baselines/b1_vision_v1.toml:32-36`），而环境每次 `step` 正好执行 `frame_skip` 个 `mj_step`（`panda_u_table_env.py:439-441`）。

因此：

```text
simulation step = 2 ms = 500 Hz
environment/control update = frame_skip * step = 2 ms = 500 Hz
steps per control period = 1
```

B1 在每个控制周期更新插值后的关节位置 command；IK 只在运动段开始时求解，不在每个 2 ms 周期内重求。

Franka 官方控制 callback 假设 1 kHz，见 `libfranka include/franka/robot.h:125-134`；`franka_ros2 v2.5.1` 的 controller manager 同样配置 `update_rate: 1000`，见 `franka_bringup/config/controllers.yaml:6-10`。当前 MuJoCo 500 Hz 不应被描述为严格复现官方时序。后续必须把所有增益、滤波和 torque-rate 限制写成显式依赖实测 `dt` 的形式，并单独决定是否需要隔离模型使用 1 ms timestep；本报告不做该参数决策。

## 6. MuJoCo 动力学接口

### 6.1 可用量和使用约束

本次对实际合并模型执行 `mj_forward`、`mj_fullM`、`mj_jacSite`、`mj_objectVelocity`、`mj_rnePostConstraint` 和 `mj_contactForce`。结果如下：

| 需求 | MuJoCo 3.10 接口 | 实际尺寸 | 结论/约束 |
|---|---|---:|---|
| 位置 | `data.qpos` | 16 | 臂关节地址为 0..6，但代码应继续使用 `jnt_qposadr` |
| 速度 | `data.qvel` | 15 | 臂 DOF 地址为 0..6；free joint 导致 `nq != nv` |
| 加速度 | `data.qacc` | 15 | forward/step 后可读 |
| command | `data.ctrl` | 8 | 语义由 actuator 决定；当前非 torque |
| actuator scalar force | `data.actuator_force` | 8 | 经 actuator 模型/限力后的 actuation-space force |
| generalized actuator force | `data.qfrc_actuator` | 15 | transmission 映射后的 generalized force |
| bias | `data.qfrc_bias` | 15 | 科氏+离心+重力，不含 passive |
| passive | `data.qfrc_passive` | 15 | spring、damper、gravcomp、fluid 之和 |
| constraint | `data.qfrc_constraint` | 15 | forward constraint solver 的 generalized force |
| inverse | `data.qfrc_inverse` | 15 | 只有调用 `mj_inverse` 后才是有效 inverse-dynamics 输出 |
| 质量矩阵 | `data.qM` + `mj_fullM(model,data,dst)` | 稀疏 65；展开 15×15 | 取 arm DOF 的 7×7 block |
| site Jacobian | `mj_jacSite` | `jacp=3×15`, `jacr=3×15` | 取 arm 列后上下堆叠为 6×7 |
| body Jacobian | `mj_jacBody`, `mj_jacBodyCom` | `jacp=3×15`, `jacr=3×15` | 分别针对 body frame 原点或 body COM；当前 CI 应优先使用 TCP site |
| TCP pose | `site_xpos`, `site_xmat` | 3、9 | 当前 `gripper_tcp` 是项目自定义 site |
| TCP twist | `Jp@qvel`, `Jr@qvel` | 3+3 | 建议显式组成 `[linear; angular]` |
| body/site 外力 | `xfrc_applied`, `cfrc_ext` | 16×6 | `cfrc_ext` 需 post-constraint RNE；空间量注意顺序 |
| contacts | `data.contact[0:ncon]` | 动态 | `mj_contactForce` 得 contact-frame `[force; torque]` |
| joint/actuator 限制 | `jnt_*`, `actuator_*limited/range` | 按对象 | 编译模型可直接查询 |

MuJoCo 3.10 官方 header 证据：

- [`mjData` fields](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/include/mujoco/mjdata.h#L246-L326)
- [`mj_rne`, Jacobian, mass, velocity, contact functions](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/include/mujoco/mujoco.h#L457-L470)
- [support functions](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/include/mujoco/mujoco.h#L570-L637)

### 6.2 `qfrc_bias` 和模型项分解

MuJoCo 的动力学约定是：

```text
qacc = M^-1 * (tau + J^T f - c)
c = Coriolis + centrifugal + gravity
```

官方依据是 [MuJoCo 3.10 computation](https://github.com/google-deepmind/mujoco/blob/28009f9105cd92784b7b0b30c0605a5e29107a77/doc/computation/index.rst#L235-L248)。`qfrc_passive` 在等式的 applied-force 一侧，不在 `c` 中；MuJoCo 3.10 还分别暴露 `qfrc_spring`、`qfrc_damper`、`qfrc_gravcomp`、`qfrc_fluid`。

本次在 home 关节位置设置非零 `dq` 后进行双评估：

1. 相同 `qpos`、`qvel=0`、原 gravity：得到 gravity。
2. 相同 `qpos/qvel`、临时 scratch 模型 gravity=0：得到 Coriolis+centrifugal。
3. 两者之和与原 `qfrc_bias` 的最大残差为 `7.1e-15 N·m`。
4. 臂 `qfrc_passive` 精确等于 `-1*dq`，符合每关节 damping=1。

可靠结论：

- 不能直接把 `qfrc_bias` 当 Franka `coriolis`。
- MuJoCo 没有一个与 Franka `model.coriolis()` 完全同名同义的现成 data field。
- 可在独立 `MjData` 中以相同 `qpos`、零 `qvel` 调 `mj_forward/mj_rne` 得 gravity，再用 `qfrc_bias-gravity` 得速度相关项。
- 也可在隔离/可恢复的模型副本上将 `opt.gravity` 设零求速度相关项，再作差得到 gravity。
- 不应在正在 step 的共享 B1 model 上临时切 gravity。
- 当前 `body_gravcomp=0`；若 torque 变体以后启用 gravcomp，则 gravity compensation 会进入 `qfrc_gravcomp/qfrc_passive`，不能再按上述控制律重复加入。

这些方法在数学结构上对应 Franka `mass/coriolis/gravity`，但数值不会严格等价：Menagerie 是简化模型，TCP/负载定义不同，抓取物通过接触约束而不是 Franka 的刚性 configured load 合入模型。

### 6.3 质量矩阵、Jacobian、坐标和存储

`mj_fullM` 展开的实际全系统矩阵是 15×15、对称；arm block 是 7×7。必须按 `arm_dof_addresses` 取块，不能假定未来模型仍为前七列。

`mj_jacSite` 返回世界坐标表达的平移 `jacp` 和旋转 `jacr`。本次探针验证：

```text
mj_objectVelocity(..., flg_local=0)[0:3] == jacr @ qvel
mj_objectVelocity(..., flg_local=0)[3:6] == jacp @ qvel
最大误差 1.39e-17
```

MuJoCo 空间速度 API 使用 `[angular; linear]`（`rot:lin`），而本报告和 Franka Cartesian impedance 统一使用 `[linear; angular]`，所以不得直接拼接 `mj_objectVelocity` 的原数组。

`site_xmat` 是扁平 3×3 rotation matrix；Python 中按 C/row-major `reshape(3,3)`。MuJoCo quaternion 是 scalar-first `[w,x,y,z]`；Eigen `Quaterniond.coeffs()` 的内存/展示惯例常为 `[x,y,z,w]`，接口处必须显式转换。当前控制代码已把 `site_xmat.reshape(3,3)` 用作世界坐标旋转，见 `panda_u_table_env.py:370-373` 和 `fixed_dls_controller.py:101-104`。

当前 TCP 是 `hand` 下 `[0,0,0.103]` 的项目 site（`panda_u_table_env.py:81-94`），并不自动等于 Franka `kEndEffector`、flange、EE frame 或 stiffness frame K。CI 前必须固定并测试这一 frame contract。

### 6.4 接触和外力

一次合法 reset 后模型有四个物体-桌面 contact；逐 contact 调 `mj_contactForce` 得到四个约 `0.24525 N` 的法向力，总和约 `0.981 N`，与 0.1 kg 物体重力一致。这证明 solver ground-truth 接触力可可靠用于仿真评测。

限制：

- 当前 `nsensor=0`，没有 wrist F/T sensor。
- `sensors/contact_sensor.py:60-65,141-160` 明确只暴露去抖后的左右指/物体布尔接触，故意隐藏位置和 solver force。
- Franka `O_F_ext_hat_K/K_F_ext_hat_K` 是由真实关节力矩传感器和机器人模型得到的滤波外力估计；`mj_contactForce/cfrc_ext` 是仿真 solver ground truth，二者不严格等价。
- 接触峰值、稳态力、穿透距离可以新增**评测诊断器**计算，不要求把 privileged solver force 暴露给控制器。

## 7. Franka 官方控制器来源

### 7.1 固定版本

| 来源 | 固定版本/commit | 文件 | 许可证 |
|---|---|---|---|
| libfranka | tag `0.21.2`，commit `9f9304ec0ac897eff3219a67f612b959948535e2`（annotated tag object `6e9b446...`） | `examples/joint_impedance_control.cpp`, `examples/cartesian_impedance_control.cpp`, `include/franka/model.h`, `include/franka/robot.h`, `include/franka/rate_limiting.h` | Apache-2.0 |
| franka_ros2 | tag `v2.5.1` / default `humble` commit `9faaaaf6ad4e4cc177cb93716547cc8ab20c5bd2` | `franka_example_controllers/src/fr3/joint_impedance_example_controller.cpp`, `joint_impedance_with_ik_example_controller.cpp`, `franka_bringup/config/controllers.yaml` | Apache-2.0 |
| MuJoCo | tag/commit `3.10.0` `28009f9105cd92784b7b0b30c0605a5e29107a77` | computation、XML reference、public headers | Apache-2.0 |
| MuJoCo Menagerie | 子模块 commit `71f066ad0be9cd271f7ed58c030243ef157af9f4` | `franka_emika_panda/panda.xml`, README | Apache-2.0 |

固定链接：

- [libfranka joint impedance](https://github.com/frankarobotics/libfranka/blob/9f9304ec0ac897eff3219a67f612b959948535e2/examples/joint_impedance_control.cpp)
- [libfranka Cartesian impedance](https://github.com/frankarobotics/libfranka/blob/9f9304ec0ac897eff3219a67f612b959948535e2/examples/cartesian_impedance_control.cpp)
- [libfranka Model API](https://github.com/frankarobotics/libfranka/blob/9f9304ec0ac897eff3219a67f612b959948535e2/include/franka/model.h)
- [libfranka Robot control API](https://github.com/frankarobotics/libfranka/blob/9f9304ec0ac897eff3219a67f612b959948535e2/include/franka/robot.h)
- [franka_ros2 joint impedance](https://github.com/frankarobotics/franka_ros2/blob/9faaaaf6ad4e4cc177cb93716547cc8ab20c5bd2/franka_example_controllers/src/fr3/joint_impedance_example_controller.cpp)

### 7.2 libfranka 关节阻抗

`examples/joint_impedance_control.cpp:174-217`：

```text
tau_i = k_i * (q_d_i - q_i) - d_i * dq_i + coriolis_i
K = [600,600,600,600,250,150,50]
D = [50,50,50,50,30,25,15]
```

输入是 `RobotState.q`, `dq`, 内部 Cartesian motion generator/IK 给出的 `q_d`（有一周期延迟）和 `Model::coriolis`；输出是七关节 `franka::Torques`。轨迹是半径 0.05 m 的圆，速度在 2 s 内 ramp 至 0.25 m/s，运行 20 s，见 `:47-57,130-171`。

安全相关：

- 先用 motion generator 到安全初始姿态（`:107-115`）。
- 显式配置碰撞阈值并要求用户手持 stop（`:117-123`）。
- 以 `limitRate(kMaxTorqueRate, ..., tau_J_d)` 限制 torque rate（`:197-213`）。
- `kMaxTorqueRate≈1000 N·m/s` 且 sample time 常量 1 ms，见 `include/franka/rate_limiting.h:18-20,41-46`。

注：该版本源码注释称 control-loop rate limiting 默认开启，但 `Robot::control` 声明实际写的是 `limit_rate=false`。本例自己显式调用 `limitRate`，因此复现应依据实际代码而不是该注释。

### 7.3 libfranka 笛卡尔阻抗

`examples/cartesian_impedance_control.cpp:33-45,67-111`：

```text
K = diag([150,150,150,10,10,10])
D = diag([2*sqrt(150)]*3 + [2*sqrt(10)]*3)
e_p = p - p_d
e_R: quaternion hemisphere correction
     q_err = q_current^-1 * q_desired
     e_R = -R_current * vector(q_err)
tau_task = J_zero^T * (-K*e - D*(J_zero*dq))
tau_cmd = tau_task + coriolis
```

它是“without inertia shaping”的固定 equilibrium spring-damper；目标是启动时 EE pose，没有目标平滑；使用 base-frame `zeroJacobian(kEndEffector)`；没有零空间项、质量矩阵 shaping、显式 torque saturation 或显式 torque-rate limit。碰撞阈值被设为很高并警告用户持有 stop（`:17-24,61-65,114-120`）。因此不能把零空间控制或目标滤波错误归因于该官方示例。

### 7.4 FCI torque command 与 MuJoCo torque 的关键差异

`libfranka include/franka/robot.h:117-134` 明确说 callback 发送“不含 gravity 和 friction”的 joint-level torque command，且频率为 1 kHz。关节示例打印时将 rate-limited command 加上 `model.gravity(state)` 后才与 measured `tau_J` 比较（`joint_impedance_control.cpp:79-93,203-209`）。

因此在 body gravcomp=0 的 MuJoCo direct motor 中，官方结构的等价候选不是简单的 `impedance+coriolis`，而是：

```text
tau_motor = impedance + coriolis_mj + gravity_mj
```

若希望额外抵消 MuJoCo 的 joint damping，则应单独处理 `-qfrc_passive`；不能把它误认为 `qfrc_bias` 的一部分。本报告不决定是否抵消该 plant damping。

### 7.5 Franka ROS 2

`franka_ros2 v2.5.1`：

- controller manager 1 kHz：`franka_bringup/config/controllers.yaml:6-10`。
- joint impedance controller 请求 7 个 `/effort` command interface，以及 7 个 position/velocity state interface：`joint_impedance_example_controller.cpp:27-47`。
- 使用平滑 cosine 目标移动关节 4/5，过滤 `dq`，输出 `K(q_goal-q)-D*dq_filtered`：`:49-67`。
- 简单版本不读取 model term；with-IK 版本读取 coriolis 并加入 command。
- 配置的 compliant gains 是 `K=[24,24,24,24,10,6,2]`、`D=[2,2,2,1,1,1,0.5]`：`controllers.yaml:66-84`。
- 当前 tag 的配置和路径明确面向 **Franka FR3**，不是 Panda；它可作为 ros2_control interface 设计参考，不能宣称数值或机器人模型等价。
- 当前官方 release 没有一个与上述 libfranka 文件同名同义的 Cartesian impedance example；有 Cartesian pose/orientation/velocity command 示例。CI 的官方算法基准应以固定 libfranka 示例为主。

## 8. 官方接口到 MuJoCo 的映射表

| Franka 官方量 | MuJoCo 候选量 | 等价等级 | 风险/差异 |
|---|---|---|---|
| `RobotState.q` | `data.qpos[arm_qpos_addresses]` | 接口等价 | 必须按 joint id/address，不能硬编码全局前七 |
| `RobotState.dq` | `data.qvel[arm_dof_addresses]` | 接口等价 | `nq != nv`；按 DOF 地址 |
| `q_d` | controller target 中的 `q_d` | 不自动提供 | MuJoCo 没有 Franka 内部 motion-generator desired state |
| `tau_J_d` | 上一周期经 rate/saturation 后的 arm command | 设计等价 | 当前 `data.ctrl` 是位置；variant 中才可用 |
| `tau_J` measured | `qfrc_actuator` 或 inverse-dynamics 诊断 | 非严格等价 | MuJoCo 没有 link-side torque sensor；actuation/solver truth 不是测量 |
| `Model::mass` | `mj_fullM` 后 arm 7×7 block | 结构等价 | 模型、payload 和接触约束不同，数值不等价 |
| `Model::coriolis` | `qfrc_bias - gravity` | 结构等价 | `qfrc_bias` 不能直接使用；含离心；需 scratch dynamics evaluation |
| `Model::gravity` | 零速度 `qfrc_bias` 或 gravity on/off 差值 | 结构等价 | 当前 gravcomp=0；不要与 gravcomp 重复 |
| `zeroJacobian(kEE)` | `vstack(mj_jacSite(jacp,jacr))[:,arm_dofs]` | 条件等价 | 都在 base/world 表达，但 TCP site 点与 Franka EE/K frame 待对齐 |
| EE position | `data.site_xpos[tcp]` | 条件等价 | 自定义 site，不是自动的 Franka EE |
| EE rotation | `data.site_xmat[tcp].reshape(3,3)` | 条件等价 | MuJoCo/Python row-major；Franka pose array column-major |
| EE twist `[lin;ang]` | `[Jp@dq; Jr@dq]` | 条件等价 | `mj_objectVelocity` 原生顺序是 `[ang;lin]`，必须重排 |
| `O_T_EE` | 由 site rotation/position 构造 4×4 | 条件等价 | 变换方向和存储格式需单测 |
| `O_F_ext_hat_K` | contact wrench / `cfrc_ext` | 不等价 | solver ground truth vs 传感器+模型估计；坐标、滤波、奇异性行为不同 |
| torque command | torque variant 的 `data.ctrl[0:7]` | 当前不等价；variant 可等价 | 依赖 motor gain=1、bias=0、gear=1 和顺序验证 |
| torque limit | actuator `forcerange/ctrlrange` | 条件等价 | 只代表仿真限制；Franka 还有硬件安全、reflex、rate limiting |
| contact state | `data.contact`, `mj_contactForce` | 仿真 ground truth | 控制器不应默认获得 privileged solver force |

## 9. JI-Baseline 定义

JI-Baseline 是基础设施和动力学正确性基线，不是主要研究改进对象。它验证：

- 七关节 torque 输入和 actuator 顺序；
- 正负方向；
- `q/dq` 索引；
- gravity、Coriolis/centrifugal 和 passive 项符号；
- timestep/control period；
- torque magnitude/rate limits；
- 基础稳定性和轨迹跟踪。

最小接口设计：

```python
tau = controller.compute(
    q=q,                 # shape (7,), rad
    dq=dq,               # shape (7,), rad/s
    target=target,       # q_d and optional dq_d
    model_terms=model_terms,
    dt=dt,
)
```

输出 `tau.shape == (7,)`，单位 N·m，且只表示机械臂；夹爪走独立接口。`model_terms` 至少区分：

```text
gravity[7]
coriolis_centrifugal[7]
passive[7]
mass[7,7]（诊断/后续使用）
tau_previous[7]
```

候选控制结构（只定义，不实现）：

```text
tau_imp = Kq (q_d-q) + Dq (dq_d-dq)
tau_raw = tau_imp + coriolis_centrifugal + gravity
tau_cmd = magnitude_limit(rate_limit(tau_raw, tau_previous, dt))
```

官方 gains 只能作为来源明确的参考点，不能视为已适配当前 500 Hz 简化模型的可用参数。进入 JI 轨迹实验前必须通过静态 gravity hold、单关节小 command 方向、零 command 下落、force clamp 和 rate-limit 单元测试。

## 10. CI-Baseline 定义

CI-Baseline 是主要研究基线，但只能建立在 JI 全部门禁通过之后。最小接口仍可保持：

```python
tau = controller.compute(
    q=q,
    dq=dq,
    target=target,       # desired TCP position/orientation, optional twist
    model_terms=model_terms,
    dt=dt,
)
```

`model_terms` 额外包含：

```text
tcp_position[3]
tcp_rotation[3,3] or normalized quaternion[4]
jacobian[6,7] with row order [linear; angular]
gravity[7]
coriolis_centrifugal[7]
tau_previous[7]
singularity diagnostics
```

忠实的 libfranka simple CI 核心应是：

```text
e = [p-p_d; e_orientation_in_world]
xdot = J dq
tau_task = J^T (-K e - D xdot)
tau_raw = tau_task + coriolis_mj + gravity_mj
```

姿态误差必须先做 quaternion hemisphere/sign correction，再采用与官方示例一致的 frame 变换；不能直接做四元数分量相减。

固定 libfranka 示例没有 zero-space term。因此第一版“官方忠实 CI”应令 `tau_null=0`。若后续增加零空间 posture control，必须作为显式扩展，分别验证：

- 投影器定义和维度；
- rank/condition number 与 damping；
- 是否使用运动学 `I-J^+J` 或动态一致投影；
- `tau_task` 与 `tau_null` 的相互污染；
- singularity 邻域连续性。

CI 还必须单独验证 TCP site 定义、直线/旋转误差、`J^T wrench` 符号、quaternion 符号连续性、冗余自由度和接触柔顺性。未完成这些验证前不能称为 Franka 等价复现。

## 11. 最小评测任务

所有任务只用于未来隔离 control harness，不接入完整抓放、Development 或 Held-out。

### A. 关节自由空间

| 实验 | 最小设计 | 主要目的 |
|---|---|---|
| A-Static | 至少四个合法初始姿态，固定 `q_d=q0` | gravity/Coriolis 符号、静态稳定 |
| A-Quintic | 每次只动一个关节，小幅五次多项式，其他关节保持 | 顺序、方向、限幅、超调 |
| A-Sine | 七关节逐个低幅低频 sine | 相位、阻尼、频率响应 |
| A-Combined | 相位错开的七关节平滑轨迹 | 耦合和总 torque |
| A-Postures | 中心姿态、近工作区边缘但远离 joint limit 的姿态 | 配置依赖和 gravity 变化 |

### B. TCP 自由空间

| 实验 | 最小设计 | 主要目的 |
|---|---|---|
| B-Axes | x/y/z 分别做小 quintic displacement | Cartesian 符号和 Jacobian |
| B-Line | 固定姿态、世界坐标直线 | 路径误差与轴耦合 |
| B-Ellipse | 固定姿态、低速闭合轨迹 | 连续跟踪与 phase lag |
| B-Rotation | 固定位置，分别绕三轴小角度 | quaternion/frame 约定 |
| B-Redundancy | 近似相同 EE pose，不同 joint posture | nullspace 与 singularity |

### C. 扰动与接触

| 实验 | 最小设计 | 主要目的 |
|---|---|---|
| C-Pulse | 用 `xfrc_applied` 或 `mj_applyFT` 施加已知短脉冲 | 恢复时间、等效柔顺 |
| C-Push | 已知恒力、结束后释放 | steady displacement、恢复 |
| C-Plane | 低速法向接触固定平面 | 峰值力、穿透、振荡 |
| C-Gain-Matrix | 少量预注册 stiffness/damping 条件 | 稳定性边界，不做自动调优 |
| C-Payload | 独立模型中改变刚性 payload，或明确使用 grasp contact | model mismatch |
| C-Model-Error | controller nominal model 与 plant model 分离 | 鲁棒性 |
| C-Timing-Noise | 控制跳步/延迟、q/dq 观测噪声 | 500 Hz 时序敏感性 |

接触实验不应从抓放开始；先使用单一平面、单一接触区域和可复核的接触法向。

## 12. 指标定义

### 12.1 轨迹和 torque 指标

对每个关节：

```text
RMSE_q_i = sqrt(mean((q_i-qdi)^2))
max_abs_q_i = max(abs(q_i-qdi))
steady_error_i = mean(error over registered final window)
overshoot_i = max signed excursion beyond final target
settling_time_i = first time after which error stays in tolerance
tau_rms_i = sqrt(mean(tau_i^2))
tau_peak_i = max(abs(tau_i))
```

TCP：

```text
position_error = ||p-p_d||2
orientation_error = ||log(R_d R^T)||2（实现时固定同一 convention）
path_error = point-to-reference-path distance
zero_space_error = ||projected posture error||
```

能由当前模型可靠直接计算：`q/dq/qacc`、TCP pose、Jacobian、command、`actuator_force/qfrc_actuator`、joint/actuator limit 触发、NaN/Inf、simulation time。

需要新增 control diagnostic logger：目标轨迹、raw/rate-limited/saturated torque、gravity/coriolis/passive 分项、condition number、各类 limit counter、energy、settling window。该 logger 只用于新 control harness。

### 12.2 接触指标

```text
peak_contact_force = max_t ||sum transformed contact forces||
steady_contact_force = mean over registered steady window
penetration = max(0, -contact.dist)
oscillation_amplitude = peak-to-peak force/displacement in steady window
recovery_time = disturbance removal to re-entry/stay in tolerance
work/energy = sum(tau^T dq * dt)
```

可靠但需新增 privileged evaluator/diagnostic：

- `mj_contactForce` 的逐 contact wrench；
- `contact.dist`；
- 接触 geom、位置、法向；
- `cfrc_ext`；
- 已知 `xfrc_applied`；
- 接触持续时间、饱和持续时间。

需要新增传感器或估计器才能与真实机器人类比：

- wrist F/T measurement；
- Franka 风格 external wrench estimate；
- link-side measured torque/noise/bias；
- 传感器带宽和滤波。

现有 `ContactSensor` 只适合布尔接触状态，不能用于 peak/steady contact force。

## 13. 稳定性与安全风险

后续实现阶段必须具备以下保护，顺序和每项触发都要记录：

1. **直接 torque 稳定性**：先 static hold 和单关节小轨迹，禁止直接进入抓放/刚性接触。
2. **仿真步长/控制周期**：当前 2 ms/500 Hz 与官方 1 ms/1 kHz 不同；所有 rate、filter、trajectory 必须按 `dt`。
3. **积分器/显式积分风险**：当前是 `implicitfast`，不是显式 Euler；它改善 velocity-dependent stiffness/damping 的数值处理，但不保证高 task stiffness、延迟控制或刚性接触稳定。未来若切换 Euler/其他积分器，必须成为单独的实验身份并重新过全部稳定性门禁。
4. **joint damping**：模型已有 `damping=1` 且在 `qfrc_passive`；不得误加到 `qfrc_bias`，是否补偿必须显式。
5. **torque magnitude**：variant 保留 `[87,87,87,87,12,12,12]` 的硬范围，并在 controller 保护层再次可观测地裁剪。
6. **torque rate**：按实际 `dt` 限制 `|tau_k-tau_{k-1}|/dt`；记录 raw 和 limited 两个值。
7. **joint position/velocity**：软限位应早于 MJCF hard limit；近限位时停止目标推进，并有速度/加速度上限。
8. **姿态误差**：限定小角度初测；用 log/quaternion error，不使用 Euler angle 差。
9. **quaternion sign**：每周期做 hemisphere continuity；归一化并拒绝零/非有限 quaternion。
10. **Jacobian singularity**：监控 singular values/condition；禁止无 damping 的伪逆；在 rank 丢失时降级或终止。
11. **零空间投影**：第一版官方 simple CI 不启用；启用前做 projector 和 task leakage 单测。
12. **接触刚度**：低速接近，设置 force/penetration/oscillation/persistent-saturation 终止条件。
13. **gravity 重复补偿**：`body_gravcomp`、controller gravity 和任何 model-side compensation 只能选择一条清晰路径。
14. **动力学符号**：以 static gravity hold 和正负 impulse 测试确认 `+qfrc_bias` 的方向，不能只凭变量名。
15. **模型误差**：分开 nominal controller model 和 plant model；payload、COM、摩擦误差单独扫，不混入 baseline 定义。
16. **异常值**：输入、model terms、raw torque、limited torque、state 和 metrics 每周期检查 NaN/Inf。
17. **紧急终止**：非有限值、越过软限位、异常速度、接触 force/penetration 超阈、连续饱和、solver warning、energy 快速增长、姿态/Jacobian 失效任一触发时，将 arm command 置为注册的安全策略并结束 episode。

保护阈值必须进入独立配置和 manifest；本报告不提供或调优阈值数值。

## 14. 对冻结 B1 的隔离方案

必须满足：

- 不改 `configs/baselines/b1_vision_v1.toml` 和 manifest。
- 不改 `PandaUTableEnv`、`load_u_table_model`、`SensorEventPickPlaceController` 和 B0/B1 注册。
- 不改 `scripts/run_development.py`、Development 协议和历史 outputs。
- 新 control runner 不接受 Development/Held-out split，也不写入正式 Development 路径。
- torque 模型、环境、配置、controller 和 outputs 使用独立命名空间。
- 新测试先运行现有 184 tests，再运行 control tests；B1 两个冻结文件 hash 和当前模型编译属性要有回归断言。
- control 实验 manifest 固定 git commit、Menagerie commit、模型 hash、MuJoCo/Python 版本、timestep、control period、integrator、seed、controller config 和保护触发。

只新增独立 model/environment 不会改变 B1，因为 B1 的入口固定加载 `models/mujoco_menagerie/franka_emika_panda/panda.xml`，见 `panda_u_table_env.py:15-24,45-97`。

## 15. 建议新增的目录和模块（本次不创建）

```text
models/franka_panda_torque/
  panda_torque.xml
  PROVENANCE.md

environments/
  panda_torque_env.py

controllers/torque/
  interfaces.py
  dynamics_terms.py
  safety.py
  joint_impedance.py
  cartesian_impedance.py

configs/control/
  ji_baseline_v1.toml
  ci_baseline_v1.toml
  experiment_matrix_v1.toml

evaluation/control/
  metrics.py
  contact_diagnostics.py
  manifest.py

scripts/
  run_control_experiment.py

tests/control/
  test_torque_model_contract.py
  test_dynamics_terms.py
  test_joint_impedance_contract.py
  test_cartesian_frames.py
  test_safety_limits.py
  test_control_smoke.py

outputs/control/            # 非正式、独立于 Development
```

`test_torque_model_contract.py` 至少检查：7 个 arm motor 的顺序、gear=1、gain=1、bias=0、ctrl/force limits、`qfrc_actuator[arm_dofs] == clipped_ctrl[0:7]`、夹爪 actuator 保持独立，以及 torque/原模型除 actuator 外的关键编译属性 parity。

## 16. 已确认事实

- 初始分支 `main`、工作区 clean；HEAD 与附件预期不同，但预期 commit 是当前 merge 的父提交。
- Menagerie 子模块 commit 与预期一致。
- 临时审计环境可运行 Python 3.12.10 + MuJoCo 3.10.0。
- 184 项测试全部通过。
- 实际模型由 Panda XML、项目 scene 和动态新增 TCP site 在内存中合并。
- `nu=8`；前七个 arm actuator、一个独立 gripper actuator。
- 环境输入是 8 维 raw actuator control，只做有限性检查和 ctrlrange clip。
- B1 IK 输出是 7 个关节位置目标，不是 torque。
- 当前七臂 actuator 是位置伺服；`data.ctrl[0:7]` 不能作为 torque。
- 当前 timestep 2 ms、control update 500 Hz、每周期一个 simulation step。
- MuJoCo 3.10 提供构建 JI/CI 所需的 state、mass、bias、Jacobian、pose、contact 和 limit 接口。
- `qfrc_bias` 是 Coriolis+centrifugal+gravity，passive 单独存放。
- 当前 body gravcomp 为零，joint damping 为 1。
- 当前没有 MJCF force/torque sensor；solver contact force 可供隔离 evaluator 使用。
- 官方 libfranka 关节/Cartesian impedance 的方程、frame、rate 和 safety 差异已有固定 commit 证据。

## 17. 尚未验证事项

- torque MJCF 尚未创建、编译或运行。
- 尚未验证 direct torque 的 actuator 顺序、符号和饱和行为。
- JI/CI 均未实现，未做任何增益调试。
- 尚未做 static gravity hold、轨迹、扰动或接触控制实验。
- 自定义 `gripper_tcp` 与 Franka `kEndEffector`/K frame 的刚体变换尚未正式登记。
- Menagerie 简化 inertial 参数与具体 Panda/末端负载的数值误差尚未量化。
- 500 Hz 和可能的 1 kHz 隔离变体之间尚未作稳定性对照。
- contact solver force 尚未封装为只对 evaluator 可见的诊断接口。
- 外力估计、link-side torque sensor、噪声和 ROS 2 runtime 均未复现。
- 未验证零空间控制；官方 simple CI 本身没有该项。
- 未访问、运行或依据 Held-out Test 作任何设计。

## 18. 最终可行性结论

```text
FEASIBLE_WITH_ISOLATED_MODEL_VARIANT
```

理由：

1. 当前模型是明确的位置伺服模型，不能满足 direct torque 输入，因此不是 `FEASIBLE_CURRENT_MODEL`。
2. Panda 的 joint、geometry、inertia、collision、scene、TCP 和全部 MuJoCo dynamics API 已经可用，没有结构性阻塞。
3. 将七臂 actuator 隔离替换成 direct motor、保留夹爪和全部其他模型内容，是小而可审计的模型变化。
4. JI 能作为 torque chain 的必要验证基线；CI 所需的 pose/Jacobian/twist/dynamics/contact 量均能得到。
5. 关键差异和风险已经明确，证据足以规划实现，不属于 `INSUFFICIENT_EVIDENCE`。

分别结论：

```text
JI-Baseline: FEASIBLE_AFTER_TORQUE_VARIANT
CI-Baseline: FEASIBLE_AFTER_JI_GATES
Contact experiments: FEASIBLE_WITH_PRIVILEGED_EVALUATOR_DIAGNOSTICS
ROS 2 mapping: INTERFACE_FEASIBLE_NOT_DROP_IN_EQUIVALENT
Current largest risk: FCI gravity/friction semantics being copied incorrectly to physical MuJoCo motor torque
```

## 19. 是否允许进入实现阶段

**有条件允许。**

下一阶段只允许：

1. 新增隔离 torque model/environment；
2. 新增模型契约和动力学项测试；
3. 新增 magnitude/rate/NaN/joint/contact 安全层；
4. 实现并运行 JI static/single-joint/free-space 基线；
5. 在 JI 全部门禁通过后，再实现 CI free-space baseline。

下一阶段仍不允许：

- 修改或复用冻结 B1 行为；
- 直接从完整抓放开始；
- 自动调参；
- 生成正式 Development 结果；
- 使用 Held-out Test；
- 宣称 Franka 等价、CI 已跑通或接触柔顺性成立。

## 20. 下一阶段建议

建议按以下 gate 顺序推进：

1. **Model Gate**：编译 torque variant；静态检查 motor/gear/gain/bias/limits/ids；确认夹爪不变。
2. **Torque Mapping Gate**：逐关节小正/负 command，检查 `actuator_force`、`qfrc_actuator`、`qacc` 的方向和饱和。
3. **Dynamics Gate**：多姿态验证 gravity、Coriolis+centrifugal、passive 分解和 mass/Jacobian 数值一致性。
4. **Safety Gate**：单测 magnitude/rate/joint/NaN/emergency stop，验证每个触发都进入 manifest。
5. **JI Gate**：A-Static 至 A-Postures 全部稳定，指标和 limit counter 完整，才允许 CI。
6. **Frame Gate**：登记 TCP↔Franka EE/K 变换，验证 pose、twist、Jacobian 和 quaternion convention。
7. **CI Free-space Gate**：B-Axes 至 B-Redundancy；先官方 simple CI（无 zero-space）。
8. **Contact Gate**：新增 evaluator-only contact diagnostics 后执行 C-Pulse 至 C-Timing-Noise。
9. **ROS 2 Mapping Gate**：最后再把经过验证的 state/model/effort interfaces 映射到选定 ROS 2/机器人版本。

本报告只证明技术路径可行并给出进入条件；它不是控制器实现、参数结论或运行成功声明。
