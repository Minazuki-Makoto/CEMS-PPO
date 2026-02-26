#CEMS + HEMS 强化学习项目文档
# 项目概述

本项目基于 PPO 强化学习 构建了多层次能源管理系统：

社区级 CEMS (Community Energy Management System)：负责社区整体能源调度与电价策略优化。

家庭级 HEMS (Home Energy Management System)：针对不同类型用户（积极、普通、消极）进行家庭设备负荷优化。

目标：在保证用户舒适度和设备约束的前提下，最大化社区和家庭回报，同时考虑可再生能源与储能系统的调度。

系统层级结构：
-Community Level (CEMS)
│
├─ Active User HEMS
├─ Normal User HEMS
└─ Inactive User HEMS
# 环境设计
## 家庭环境

| 环境类型      | 状态维度 | 动作维度 | 描述                         |
|---------------|----------|----------|------------------------------|
| active_env    | 8        | 8        | 积极响应用户，偏向调节负荷以降低成本 |
| normal_env    | 8        | 7        | 普通用户，适度调整负荷        |
| inactive_env  | 5        | 5        | 消极用户，对负荷控制反应较低  |

家庭环境主要考虑：

室内温度、湿度、电池 SOC（状态电量）

空调功率、可控负荷设备功率

用户舒适度损失、异常惩罚

## 社区环境 (community_env)

| 环境类型      | 状态维度 | 动作维度 | 描述                                 |
|---------------|----------|----------|--------------------------------------|
| active_env    | 8        | 8        | 积极响应用户，偏向调节负荷以降低成本 |
| normal_env    | 8        | 7        | 普通用户，适度调整负荷                |
| inactive_env  | 5        | 5        | 消极用户，对负荷控制反应较低          |
| community_env | 4        | 2        | 社区层面控制，协调多类用户负荷        |

##社区环境负责：

汇总各家庭负荷

根据社区电价策略调整能源分配

对社区奖励进行计算，包括总成本、储能管理和功率平滑

# 强化学习智能体设计
## 家庭级 PPOHEMS_agent

输入：家庭状态 state_dim

输出：家庭动作 action_dim

隐藏层维度：hidden_dim = 256

奖励函数结合：

电价成本

用户舒适度

SOC 越界惩罚

## 社区级 PPO_agent

输入：社区状态 community_state_dim = 4

输出：社区价格策略 community_action_dim = 2

奖励函数：

汇总社区总回报

储能系统管理

功率平滑与惩罚

# 训练流程
## 初始化阶段

设置随机种子：
# 随机种子设置

##1.为了保证实验的可复现性，需要设置 Python、NumPy 和 PyTorch 的随机种子。

```python
SEED = 64
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
```
##2.生成 初始负荷曲线：
```python
for alt in range(11000):
    active_environment.reset()
    normal_environment.reset()
    inactive_environment.reset()
    ...
```
对每类用户运行 11000 次仿真

保存最优回报对应的负荷策略 P_all

## 社区迭代训练

外循环：12 次
内循环：

社区 CEMS 训练 20000 次

根据社区最优价格生成 price_h

各类型家庭环境根据 price_h 训练 HEMS PPO：

5000 次更新

记录每个用户类型的最优回报和功率

更新社区总负荷 CM_P_all

## 收敛判定

当连续两次 CEMS 总回报变化小于 0.5*50 时停止训练

# 奖励函数设计
## 家庭级奖励
```python
R_hems = - (电费 + 舒适度损失 + SOC 越界惩罚)
```
电费：根据电价和负荷计算

舒适度损失：温度/湿度偏离理想值

SOC 越界惩罚：储能状态超出上下限时的惩罚

## 社区级奖励
``` python
R_cems = Σ(m*R_active + n*R_normal + p*R_inactive) - 储能惩罚 - 功率平滑惩罚
```
# 数据保存与可视化
## 保存策略 CSV

用户最优策略：
``` python
D:/pycharmcode/project/CEMS/optimal_strategy_active.csv
D:/pycharmcode/project/CEMS/optimal_strategy_normal.csv
D:/pycharmcode/project/CEMS/optimal_strategy_inactive.csv
```
社区策略：
``` python
D:/pycharmcode/project/CEMS/optimal_strategy_community.csv\
```
包含时间、电价、社区功率和储能剩余

## 可视化示例
``` python
plt.figure(figsize=(18,16))
plt.grid(True)
plt.title('HEMS Reward over Iterations')
plt.plot(range(len(hems_reward_history)), hems_reward_history, lw=1.5, color='blue', marker='o')
plt.show()

plt.figure(figsize=(18,16))
plt.grid(True)
plt.title('CEMS Reward over Iterations')
plt.plot(range(len(cems_reward_history)), cems_reward_history, lw=1.5, color='red', marker='s')
plt.show()
```

# 总结

构建了 多层次强化学习能源管理系统

社区层 CEMS 协调价格和储能，家庭层 HEMS 优化设备调度

支持三类用户行为建模（积极、普通、消极）

使用 PPO 算法进行策略优化

可视化和 CSV 保存结果，便于后续分析和部署


