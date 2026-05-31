# UAV Multi-Modal Trajectory Prediction

这是一个面向 ProjectAirSim 无人机运动数据的轻量 3D 多模态轨迹预测框架，服务于背景无人机行为建模、对抗测试和控制算法闭环验证。模型借鉴 TNT / DenseTNT 的目标点条件化思路，但不依赖地图、车道线等接口：

1. 以 `H` 帧位置、四元数、线速度、角速度和线加速度编码运动历史。
2. 预测 `K` 个 3D 未来终点位移候选。
3. 针对每个终点补全一条 `F` 帧未来轨迹。
4. 对候选轨迹评分并按概率从高到低输出。

模型内部预测相对于当前位置的位移；离线/在线导出结果会恢复为 ProjectAirSim 原生 NED 绝对坐标。

## Structure

```text
uav_multimodal_prediction/
  config.py
  data/                  # dataset adapter, normalizer, collate
  models/                # GRU encoder, predictor, losses, metrics
  online/                # cold-start buffer, ProjectAirSim adapter
  utils/                 # geometry, checkpoints, logging
train.py
infer.py
visualize_trajectory.py
```

## Installation

```bash
python -m pip install -r requirements.txt
```

在线推理还需要根据官方仓库安装 ProjectAirSim Python client：

```bash
python -m pip install -e /path/to/ProjectAirSim/client/python/projectairsim
```

## Dataset

数据由 ProjectAirSim 仿真采集脚本生成，数据收集仓库见 [QinCheng0928/ProjectAirSim-UAV-Kinematic-DataGen](https://github.com/QinCheng0928/ProjectAirSim-UAV-Kinematic-DataGen.git)。

当前训练默认使用已发布到 Hugging Face 的数据集：[qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories](https://huggingface.co/datasets/qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories)。训练脚本通过 `datasets` 加载：

```python
from datasets import load_dataset
ds = load_dataset("qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories")
```

## Training

```bash
python train.py 
  --dataset_name qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories 
  --output_dir outputs/uav_dense_goal 
  --history_len 20 
  --future_len 30 
  --num_modes 6 
  --batch_size 64 
  --epochs 20 
  --hidden_dim 16 
  --stride 1 
  --device auto
```

训练目标为 best-of-K / winner-takes-all：

```text
loss = SmoothL1(best_trajectory, gt)
     + lambda_cls * CrossEntropy(score_logits, best_mode)
     + lambda_goal * FDE(best_goal, gt_final_position)
```

日志包含 `loss`、`reg_loss`、`cls_loss`、`goal_loss`、`minADE`、`minFDE`、`MR`、`Top1 ADE/FDE`。checkpoint 包含模型、优化器、配置和训练集标准化统计。

推理、可视化和在线节点通过项目内的 `load_checkpoint()` 加载权重，默认使用 `torch.load(..., weights_only=True)`。

## Offline Inference

```bash
python infer.py 
  --checkpoint outputs/uav_dense_goal/checkpoint_epoch_0020.pt 
  --split test 
  --num_samples 50 
  --output outputs/predictions.json 
  --format json
```

保存 NPZ：

```bash
python infer.py --checkpoint outputs/uav_dense_goal/checkpoint_epoch_0020.pt 
  --output outputs/predictions.npz --format npz
```

安装 `matplotlib` 后可以添加 `--plot_index 0` 显示一个样本的 3D 预测。

## Trajectory Visualization

专用可视化脚本会画出 `history_len + future_len` 个真实轨迹点，并从历史末点开始画出 `num_modes` 条预测轨迹：

```bash
python visualize_trajectory.py \
  --checkpoint outputs/uav_dense_goal/checkpoint_epoch_0020.pt \
  --split test \
  --sample_index 0 \
  --output outputs/trajectory_visualization.png
```

默认只从完整历史窗口中取样，避免把冷启动虚拟历史误当作真实轨迹。若确实想检查冷启动窗口，可添加 `--include_cold_start`。

## ProjectAirSim Online Inference

ProjectAirSim 适配器使用官方接口：

```python
client = ProjectAirSimClient()
client.connect()
world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
drone = Drone(client, world, "Drone1")
kinematics = drone.get_ground_truth_kinematics()
```

在线节点保持 ProjectAirSim 的 NED 坐标系和弧度约定，并以 `asyncio` 执行固定频率循环：

```bash
python -m uav_multimodal_prediction.online.predictor_node \
  --checkpoint outputs/uav_dense_goal/checkpoint_epoch_0020.pt \
  --scene_config scene_basic_drone.jsonc \
  --drone_name Drone1 \
  --frequency_hz 10 \
  --log_path outputs/online_predictions.jsonl
```

每周期读取 ground-truth kinematics，更新历史缓冲，预测 top-K 轨迹，打印终点/概率并保存 JSONL。不包含飞控闭环控制命令，可作为后续 RL 环境或对手策略模块的预测插件。

## Cold Start

`HistoryBuffer` 允许模型在启动瞬间立刻工作：

1. `reset(start_state)` 以起点位置、默认姿态 `[0, 0, 0, 1]`、零速度/角速度/加速度填充全部 `H` 帧。
2. 每到达一个真实状态，`update(real_state)` 将最旧的虚拟或真实状态滑出，把新状态放在窗口尾部。
3. `get_history()` 返回历史及 `[H]` mask：虚拟点为 `0`，真实点为 `1`。
4. `real_count` 与 `is_fully_warmed` 可用于监测预热进度；模型将 mask 作为编码输入的一部分。

离线数据集采用同样的冷启动策略：每条 episode 会额外生成真实历史长度为 `1` 到 `H - 1` 的窗口，左侧虚拟帧保留 episode 起始位置和姿态，同时将速度、角速度和加速度置零；对应 `history_mask` 中虚拟帧为 `0`、真实帧为 `1`。因此模型在训练阶段可以见到在线启动时会出现的历史分布。

## Output Format

离线及在线对外结果均为绝对 NED 坐标：

```text
trajectories: [K, F, 3]
scores:       [K]
goals:        [K, 3]
```

模型原始 `forward()` 返回相对位移版本 `trajectories` 与 `goals`；传入 `current_position` 时同时返回 `absolute_trajectories` 和 `absolute_goals`，方便独立接入其他仿真或强化学习环境。
