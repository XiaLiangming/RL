# Play Atari Breakout with Deep Q-Learning

## 项目简介

使用 Deep Q-Learning 来训练 Atari 中 Breakout 游戏的控制策略

## 依赖

通过运行以下代码来快速构建环境：

```
conda env create -f environment.yml
conda activate cartpole_env
```

## 文件结构

```
DQN_Breakout/
├─ gifs            # 可视化的训练结果
├─ history         # 训练过程记录
├─ models          # 训练后的模型权重
├─ plots           # 训练过程中的评估奖励曲线
├─ buffer.py       # 存放经验缓冲区类（ER、PER）
├─ dqn_double.py   # Double DQN
├─ dqn_nature.py   # 2015 Nature DQN
├─ dqn_vanilla.py  # 2013 Vanilla DQN
├─ env.py          # 存放 Breakout 环境类
├─ environment.yml # 依赖
├─ model.py        # 存放 Pytorch DQN 模型类（DQN、Dueling DQN）
├─ README.md       # 本文件
└─ utils.py        # 存放评估函数
```