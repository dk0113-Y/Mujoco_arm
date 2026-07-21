# B1-Vision v1 冻结报告

## 1. 冻结结论与定义

B1-Vision v1 Freeze Verification 在 commit `bf6a07945f396f7b98f5c24cf94d1a97b8dc7f9d` 上通过。当前状态是 `verified_pending_user_commit`：行为配置已验证并生成冻结包，但本任务没有自动 commit、tag 或 push，`final_git_commit` 保持 `null`。

B1-Vision v1 是后续 B2、残差学习和独立强化学习方法的固定工程基线。

它不是生产部署系统，不代表已验证泛化、真机水平或最优性能，也不表示视觉优于 Oracle。

## 2. 方法、传感器与权限边界

B1-Vision v1 使用 `SensorEventPickPlaceController`、固定阻尼 DLS、`ControllerConfig`、`B1Config` 和传统 RGB-D 感知。输入包括固定俯视相机的 RGB-D、本体关节/TCP 状态、夹爪编码器和由 MuJoCo 接触对构造的双指 simulated tactile proxy。

在线控制不得读取 object body/site 真值、target site 真值、最终任务成功标签、未来状态、接触力、最优路径或 IK 可达性标签。初始定位、预抓取重定位和最终视觉验证来自 RGB-D；下降、试抬、搬运、释放和撤离由状态机、本体与夹爪/接触事件驱动。独立 privileged evaluator 只在结果记录阶段计算 ground-truth success、false positive 和 false negative，不向控制器反馈。

B0-Oracle 与 B1-Vision 使用完全相同的 controller class、Fixed-DLS、控制配置、状态机、hold、timeout、抓取几何、夹爪反馈和接触传感器；唯一方法差异是外部物体/目标位置来自有限 Oracle provider 或 RGB-D provider。

## 3. 配置与代码 provenance

| 项目 | 值 |
|---|---|
| 验证 commit | `bf6a07945f396f7b98f5c24cf94d1a97b8dc7f9d` (`Codex Round 0.5`) |
| 验证分支 | `freeze/b1-vision-v1` |
| submodule | `mujoco_menagerie@71f066ad0be9cd271f7ed58c030243ef157af9f4` |
| 源模板 | `configs/baselines/b1_vision_calibration_template.toml` |
| frozen config | `configs/baselines/b1_vision_v1.toml` |
| 两个配置 SHA-256 | `6808c142ae8805695fc43d5e4743a9529cdbea15008810456184e40e1c4b7ea9` |
| 行为等价 | 文件字节相同；加载后的完整 `EnvConfig`、`ControllerConfig`、`B1Config` 相同 |

配置 schema 没有独立、可确认不参与 loader 行为的 frozen metadata，因此没有向 TOML 强行加入字段。冻结状态只记录在 `configs/baselines/b1_vision_v1_manifest.json`。

## 4. Evaluation Protocol 与 Calibration split

| 项目 | 值 |
|---|---|
| Protocol | `evaluation_protocol/1.0.1` |
| metrics schema | `1.0.0` |
| protocol SHA-256 | `7a47be9ddf3851b06c84068ec29030d5bf25ebf60f37057d55371823b07e10bd` |
| split ID | `evaluation_protocol_v1` |
| Calibration | 30 个唯一 seed，SHA-256 `1a92bfc8f5dfce78883bc92b7e4c70f66491eb8c067a41c88fd011506505ec7e` |
| Development | 60 个 seed，未运行 |
| Held-out Test | 100 个 seed，未使用 |

正式任务保持 random pick、random place、random mass/friction、35 s episode timeout、原始成功条件和碰撞规则。protocol TOML、三个正式 seed 文件与 split manifest 均未修改。

## 5. Round 0 与 Round 0.5

Round 0 在同一模板、protocol 和 Calibration split 上完成 30/30 pair：B0 safe success 20/30，B1 safe success 17/30，collision 0/30，unexplained failure 0/30。其 15 个文件的规范哈希集合为 `072beba34bbbd88288b4169c8fd31306db424713fbb38115bbbbd5b01ac0a50a`。

Round 0.5 对 seeds 2802、3915、2957、1268 做了被动 instrumentation replay。8/8 episode 的 fingerprint、终态、失败原因、controller/GT success、碰撞和仿真时间与 Round 0 一致；privileged diagnostic 数据没有进入控制器。structured review 选择决策 B：没有明确 Calibration 参数问题，不运行 Round 1，也不放宽 aperture drop、hold 或 timeout。

诊断证据显示主要失败是固定抓取几何和接触敏感性：典型失败有明显 edge/tipped grasp、9–21 mm 相对滑移、约 1.58 rad 倾斜与远超 3 mm 的 aperture loss。放宽 aperture threshold 会把真实不稳定抓取伪装为稳定抓取，因此没有参数调整依据。Round 0.5 的 93 个既有文件仍与其 manifest artifact hashes 一致；artifact set SHA-256 为 `299d9bc626535c1a974ecf8cdf57946b82b5da6bf59945ca8a91006a185501b6`。

## 6. Freeze Verification 执行与完整性

正式命令：

```powershell
& C:\ai_workspace\venvs\Mujoco_arm\Scripts\python.exe scripts\run_calibration.py `
  --protocol configs\protocols\evaluation_protocol_v1.toml `
  --baseline-config configs\baselines\b1_vision_calibration_template.toml `
  --seeds-file configs\splits\evaluation_protocol_v1\calibration_v1.txt `
  --output-dir outputs\calibration\b1_vision_v1\freeze_verification `
  --require-clean-git
```

运行开始时 Git 工作区干净，runner 记录 commit `bf6a079`、无 effective override、`calibration_run=true`、`automatic_parameter_search=false`。正式 runner 没有 diagnostic factory 参数，因此 controller 的 `diagnostic_observer=None`；没有启用 diagnostics 或 visualization，没有生成 trace/frame 文件，正式 EpisodeResult/CSV schema 中也没有 `diagnostic.*` 或 `privileged_diagnostic.*` 字段。

完整性结果：requested/completed pair = 30/30；B0/B1 episode = 30/30；invalid pair = 0；program error = 0；invalid numeric = 0；seed 唯一且无缺失；所有 pair fingerprint 有效；config/protocol/split hash 与 Round 0 相同。

## 7. 逐 seed 行为一致性

`scripts/verify_b1_freeze.py` 是纯读取比较器：它不运行 controller、不修改配置，只读取 Round 0、Freeze Verification、Calibration split、protocol 和 frozen config candidate。它明确不读取 Development 或 Held-out Test。

全部 60 个 seed×method episode 的以下字段逐项一致：pair ID、fingerprint、pick/place 位置与区域、mass、三维 friction、final stage、failure reason、controller success、privileged GT success、placement/safe success、collision count/episode、false positive/negative、unexplained failure 和 program error。全部 30 个 pair 的 outcome category 与成对字段一致。

必须精确一致的字段采用精确比较。`simulation_time` 仅使用 Round 0.5 已验证的规则：`rtol=0`、`atol=0.0020000001 s`；本次最大实际差值是 `0.0 s`，没有放宽规则。failure reason counts、final stage counts、production metrics 和 cycle-time 汇总也一致。

比较产物：

- `outputs/calibration/b1_vision_v1/freeze_verification/freeze_comparison.json`
- `outputs/calibration/b1_vision_v1/freeze_verification/freeze_comparison.csv`
- `outputs/calibration/b1_vision_v1/freeze_verification/freeze_verification_report.md`

## 8. 五个生产型指标与周期时间

| 方法 | safe task success | first-attempt placement | collision episode | unexplained failure | safe-success simulation time median |
|---|---:|---:|---:|---:|---:|
| B0-Oracle | 20/30 | 20/30 | 0/30 | 0/30 | 18.729000 s |
| B1-Vision | 17/30 | 17/30 | 0/30 | 0/30 | 18.758000 s |

B1 placement success 17/30；controller-reported success 17/30；privileged ground-truth success 17/30；false positive 0；false negative 0。B1 safe-success 仿真周期时间：count 17，mean 18.733765 s，minimum 18.298000 s，maximum 19.080000 s。仿真时间是任务流程代理，不等于 wall-clock latency 或真实工业节拍。

Pair 分类为 both_success 16、oracle_only_success 4、vision_only_success 1、both_failed 9。这个结果不支持“视觉优于 Oracle”的结论；B0 只是有限真值外部状态的诊断对照，也仍受同一抓取、接触、IK、碰撞与释放机制限制。

## 9. 失败类型、能力边界与后续方向

B1 failure reason 为 success 17、grasp_not_confirmed 7、initial_perception_failed 3、pregrasp_reacquisition_failed 2、grasp_lost_during_transfer 1。主要工程限制是固定抓取点/接近姿态在接触几何上的敏感性，以及部分视角下初始/预抓取感知失败。

当前能力只覆盖单个固定尺寸红色立方体、单个静态绿色目标、单台固定俯视相机、传统阈值 RGB-D 感知、固定 DLS 和 simulated tactile proxy。不包含连续视觉伺服、多物体/多目标、形状随机化、动态目标、自适应 DLS、零空间控制、神经网络感知、完整重抓流程或真实硬件触觉标定。

后续应在 Development 60 上开展 B2 设计，重点研究 geometry-aware centering、接近姿态鲁棒性和能区分持续边缘接触与稳定中心抓取的 contact/pose-aware grasp quality。不得反向修改冻结的 B1-Vision v1，也不得用 Held-out Test 调参或选择 B2。

## 10. 冻结边界与待办

- 未运行 Round 1。
- 未修改任何 B1、controller、perception、camera、environment、physics、success 或 protocol 参数。
- 未运行 Development 60。
- 未运行或查看 Held-out Test。
- 未实现 B2。
- 冻结不代表生产部署、工业安全认证、真机验证、泛化证明或最终算法排名。
- 用户最终 commit/tag 尚未完成；应先审查配置、manifest、报告、比较工具和测试，再决定最终 commit 与 tag。

