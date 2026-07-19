# MuJoCo Panda U 形工作台环境

本项目当前提供一个统一、可配置、可复现的 Franka Panda 抓取与放置环境。模型使用 MuJoCo 3.10.0 和只读的 `mujoco_menagerie` Panda submodule；Task 1 控制器直接读取 MuJoCo 仿真真值，不包含摄像头或视觉估计。

## U 形工作台

三块独立、轴对齐的 box geom 构成开口朝 `-X` 的 U 形工作台，桌面顶部统一为 `z = 0.22 m`：

| 区域 | body / geom | 中心 `(x, y, z)` m | 完整尺寸 `(x, y, z)` m |
| --- | --- | --- | --- |
| front | `u_table_front` / `u_table_front_geom` | `(0.55, 0.00, 0.20)` | `(0.60, 0.90, 0.04)` |
| left | `u_table_left` / `u_table_left_geom` | `(0.10, 0.57, 0.20)` | `(0.90, 0.24, 0.04)` |
| right | `u_table_right` / `u_table_right_geom` | `(0.10, -0.57, 0.20)` | `(0.90, 0.24, 0.04)` |

Panda 基座位于原点。默认基座清空半径为 `0.20 m`；采样还计入物体半尺寸。每个区域的有效 XY 范围由已编译模型中的 geom 尺寸、位置和配置的 `edge_margin` 推导，不依赖 Fixed-DLS 是否能到达。

## 配置与复现

默认配置是 `configs/u_table.toml`，包括：

- `pick.mode`：抓取点 `fixed` / `random`；
- `place.mode`：放置点 `fixed` / `random`；
- `physics.mode`：物体质量和三项 MuJoCo friction `fixed` / `random`；
- 全局 seed、允许区域、桌边安全距离、抓放最小 XY 距离；
- settle time、episode timeout、frame skip、Viewer 默认值；
- 原 Fixed-DLS 阻尼、步长、误差阈值、waypoint、动作时长和夹爪控制量。

环境只拥有一个 `numpy.random.Generator`。显式调用 `reset(seed=42)` 会重建该 generator，因此抓取位置、目标位置、质量、摩擦和 reset 后状态可复现。几何合法样本不会因为 Fixed-DLS 不可达而重采样；这类情况会作为控制器失败记录。

## 运行

从仓库根目录使用项目虚拟环境：

```powershell
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --headless
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --viewer
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\view_environment.py --config configs\u_table.toml --seed 42
```

命令行可用 `--pick-mode`、`--place-mode` 和 `--physics-mode` 覆盖三类 mode。完整运行会输出人类可读摘要及 JSON `EpisodeResult`。退出码为：成功 `0`、可解释的控制器/任务失败 `2`、未预期程序错误 `1`。仅验证 reset 而不打开 GUI 时，可为查看入口添加 `--headless`。

## 环境接口与观测

`PandaUTableEnv` 提供 `reset(seed=None, options=None)`、`step(control)`、`observation()`、`success()`、`failure_reason()` 和 `close()`。观测包括臂关节位置/速度、手指位置、TCP 位置/姿态、仿真时间，以及：

- `privileged_object_position`
- `privileged_place_target_position`

这两个字段是 Task 1 中控制器使用的仿真真值（privileged observation），不是 RGB、深度或视觉算法的估计结果。

## 当前控制基线与边界

控制器保留原脚本的固定阻尼 DLS 公式、世界系位姿误差、Jacobian、smoothstep 关节目标插值、夹爪开合与抓放状态机。它能结构化报告 reset、IK、waypoint、桌面碰撞、抬升、掉落、放置误差、超时及未预期异常。

当前不包含自适应 DLS、零空间关节限位优化、实时任务空间闭环改进、强化学习、模仿学习或任何学习框架。合法随机区域中的部分任务可能超出这一基线控制器的能力；失败会保留原样并记录，而不是通过重采样隐藏。Task 2 将增加固定俯视 RGB-D 感知，并把视觉 observation 与当前 privileged truth 明确区分。

## 测试

```powershell
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe -m compileall environments controllers evaluation scripts tests
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe -m unittest discover -s tests -v
```
