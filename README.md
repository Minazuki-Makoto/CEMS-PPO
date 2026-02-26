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
环境类型	  状态维度	   动作维度	         描述
active_env    8	       8	   积极响应用户，偏向调节负荷以降低成本
normal_env	  8	       7	     普通用户，适度调整负荷
inactive_env  5	       5	    消极用户，对负荷控制反应较低

家庭环境主要考虑：

室内温度、湿度、电池 SOC（状态电量）

空调功率、可控负荷设备功率

用户舒适度损失、异常惩罚
