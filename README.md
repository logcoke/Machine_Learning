# Machine_Learning

> 一个面向系统化学习与实践经典机器学习算法的仓库，主要采用 Jupyter Notebook 展示推导、实现与实验对比。  
> This repository provides structured notebooks and reusable Python utilities for learning, implementing, and experimenting with classical machine learning.

## 目录目标 (Purpose)
本仓库旨在：
- 梳理常见监督 / 非监督学习算法的核心思想与数学推导
- 提供可运行的最小可行实现与对比实验
- 积累特征工程、模型评估、调参方法
- 构建个人可迭代的学习与实践路线

## 技术栈 (Stack)
- 主要语言：Jupyter Notebook (≈95.8%), Python (≈4.2%)
- 推荐依赖：`numpy` `pandas` `scikit-learn` `matplotlib` `seaborn`
- 后续可扩展：`xgboost` `lightgbm`、`pytorch` / `tensorflow`（深度学习部分）

## 初始规划的目录结构 (Planned Structure)
```
Machine_Learning/
├── notebooks/                # 交互式算法与实验笔记
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_svm.ipynb
│   ├── 04_tree_models.ipynb
│   ├── 05_clustering.ipynb
│   ├── 06_dimensionality_reduction.ipynb
│   ├── 07_neural_networks_intro.ipynb
│   └── model_evaluation.ipynb
├── src/                      # 可复用函数与工具
│   ├── data_preprocessing.py
│   ├── metrics.py
│   ├── visualization.py
│   └── model_wrappers.py
├── datasets/                 # 小型示例数据（或下载脚本）
│   └── README.md
├── experiments/              # 模型对比 / 调参记录
│   └── hyperparameter_search.ipynb
├── docs/                     # 理论笔记与公式
│   ├── loss_functions.md
│   ├── optimization_algorithms.md
│   └── feature_engineering.md
├── requirements.txt
├── LICENSE (待选择)
└── README.md
```

## 快速开始 (Quick Start)
```bash
# 克隆仓库
git clone https://github.com/logcoke/Machine_Learning.git
cd Machine_Learning

# 安装依赖（建议使用虚拟环境）
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook
```

## 学习路径建议 (Recommended Learning Path)
1. 线性回归：最小二乘、正则化 (L1/L2)  
2. 逻辑回归：概率解释、决策边界、交叉熵  
3. 支持向量机：最大间隔、核函数思想  
4. 树模型与集成：决策树、随机森林、梯度提升  
5. 聚类与降维：K-Means、层次聚类、PCA、t-SNE  
6. 特征工程与数据预处理：编码、标准化、特征选择  
7. 模型评估：交叉验证、ROC/AUC、混淆矩阵、学习曲线  
8. 神经网络基础（可选扩展）：前向 / 反向传播、激活函数  
9. 超参数优化：网格搜索、随机搜索、贝叶斯优化（后续）  

## 实验规范 (Experiment Conventions)
每个实验 Notebook 推荐包含：
- 背景说明
- 数据集描述与来源
- 模型/算法选择理由
- 评估指标与可视化
- 结果分析与改进方向
- 复现实验所需的随机种子与环境说明

## 计划 (Roadmap)
- [ ] 添加线性/逻辑回归推导 Notebook
- [ ] 建立数据预处理工具模块
- [ ] 模型评估与可视化统一接口
- [ ] 集成模型对比实验（树 / 集成 / 线性）
- [ ] 添加特征工程文档
- [ ] 加入超参数搜索示例
- [ ] 深度学习入门：简单全连接网络
- [ ] 实验结果持久化（CSV / MLflow 任选）
- [ ] 增补 LICENSE
- [ ] 添加中文 + 英文双语索引

## 推荐 Topics (可添加到 GitHub)
`machine-learning`, `data-science`, `jupyter-notebook`, `python`, `scikit-learn`, `regression`, `classification`, `clustering`, `feature-engineering`, `model-evaluation`, `tutorial`, `notebooks`, `visualization`

后续增加深度学习内容后可再添加：`deep-learning`, `neural-networks`, `pytorch`, `tensorflow`

## 贡献 (Contributing)
欢迎提交：
- 改善算法推导的笔记
- 增加更具代表性的数据集示例
- 新的评估/调参方法
- 性能或可读性优化

提交前建议：
1. 保持 Notebook 的执行顺序清晰（Kernel 重启后可完整运行）  
2. 避免提交大型数据集（> 10MB）——可使用下载脚本  
3. 使用统一代码风格（建议 `black` + `ruff`）  

## 环境与风格 (Environment & Style)
- 版本控制：固定核心库版本以保证复现
- 编码规范：PEP 8（自动化格式化）
- 随机性：统一设置 `random_state` 保障结果可复现

## License
尚未选择。可考虑：
- MIT：最宽松，适合学习示例
- Apache-2.0：专利友好
- CC-BY-4.0：若偏重文档与教程内容

## 更新日志 (Changelog)
- v0.0.1：初始化 README 草稿

## FAQ
Q: 为什么不用深度学习开头？  
A: 先掌握传统机器学习的统计与建模基础，再自然过渡到更复杂的网络结构。  

Q: 数据集来自哪里？  
A: 初期使用公共经典数据集（Iris、Boston Housing、Wine 等），后期可添加下载脚本或 Kaggle 链接。  

---

欢迎提出 Issue 或直接发起 Pull Request 共同完善。  

Enjoy Learning & Experimenting!
