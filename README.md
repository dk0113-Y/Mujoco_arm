# MuJoCo Panda U 形工作台 RGB-D 环境

本项目提供一个统一、可配置、可复现的 Franka Panda 抓取与放置环境。Task 2 在 Task 1 的 U 形工作台和固定阻尼 DLS 基线上增加了固定俯视 RGB-D 相机、传统颜色/深度检测、三维反投影，以及 `privileged` / `perception` 任务状态源切换。

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

两者使用稳定 ID `pick_object_0` 和 `place_target_0`，并复用同一个 Fixed-DLS waypoint/状态机。控制器文件不读取 object body、target site 或 `current_episode`；只有独立 evaluator 在感知估计产生后读取同一时刻真值计算误差。perception 模式的环境 observation 和 reset info 也不会暴露外部任务真值。

感知失败不会回退到 privileged provider，可能返回：

- `perception_object_not_found`
- `perception_target_not_found`
- `perception_invalid_depth`
- `perception_projection_error`
- `perception_low_confidence`

## 配置与复现

默认配置为 `configs/u_table.toml`，包括 pick/place/physics 的 fixed/random 模式、全局 seed、工作区安全边距、仿真参数、相机参数、感知阈值和原 Fixed-DLS 参数。

环境只维护一个 `numpy.random.Generator`。显式 `reset(seed=42)` 会重建 Generator，使抓取位置、目标位置、质量、摩擦和 reset 状态可复现。合法几何样本不会因为控制或视觉失败而重采样。

## 运行

```powershell
# privileged 对照
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --observation-source privileged --headless

# RGB-D perception
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --observation-source perception --headless

# Viewer
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_pick_place.py --config configs\u_table.toml --seed 42 --observation-source perception --viewer

# 捕获 RGB、metric depth、mask、NumPy 原始数组和相机元数据
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\capture_rgbd.py --config configs\u_table.toml --seed 42 --output-dir outputs\perception_debug

# 默认以 random pick/place 评测多个 seed
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\evaluate_perception.py --config configs\u_table.toml --seeds 42 43 44 45 46
```

`capture_rgbd.py` 输出 `rgb.png`、16-bit 毫米深度、8-bit 深度预览、两个 mask、原始 `.npy` 和 `metadata.json`。`outputs/` 已被 Git 忽略。

完整运行输出人类摘要和 JSON `EpisodeResult`。退出码为成功 `0`、可解释失败 `2`、未预期程序错误 `1`。

## 当前控制基线和能力边界

Fixed-DLS 公式、固定阻尼、世界系位姿误差、Jacobian、关节步长限制、smoothstep 和抓放状态机均未改变。机器人自身的 qpos、qvel 和 TCP 状态仍可直接使用。

当前只有单个固定尺寸立方体、单个静态目标和单台固定相机。抓取时夹爪可能完全遮挡立方体；重复感知会诚实返回 `perception_object_not_found`，不会使用真值补救。当前不包含神经网络视觉、形状随机化、多物体、多目标、动态目标、自适应 DLS、零空间优化、模仿学习或强化学习。

## 测试

```powershell
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe -m compileall environments controllers evaluation perception scripts tests
C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe -m unittest discover -s tests -v
```
