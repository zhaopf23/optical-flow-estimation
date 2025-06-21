# 光流估计项目
本项目使用深度学习技术实现光流估计，基于 Flying Chairs (https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)数据集训练。

## 功能
- 数据加载与预处理
- FlowNet 网络实现
- 多尺度损失函数
- 学习率调度器

## 安装
```bash
git clone https://github.com/zhaopf23/optical-flow-estimation.git
cd optical-flow-estimation
pip install -r requirements.txt

## 项目结构
flow-project/
├── src/ - 源代码目录
│ └── train.py - 主训练脚本
├── FlyingChairs_release/ - 数据集目录（需自行下载）
├── .gitignore - Git忽略规则
├── requirements.txt - Python依赖
└── README.md - 项目文档

