# Q-Learning CartPole-v1

## 项目简介

使用表格型 Q-Learning 来训练 Gymnasium 的 CartPole-v1 环境的控制策略。

## 依赖

通过运行以下代码来快速构建环境

```
conda env create -f environment.yml
conda activate cartpole_env
```

## 文件结构

```
Q-Learning_CartPole_v1/
├─ environment.yml # 依赖
├─ history.csv     # 训练历史
├─ history.png     # 训练过程曲线
├─ main.py         # 主脚本，包含训练/评估逻辑
├─ README.md       # README
└─ result.gif      # 可视化的训练结果
```