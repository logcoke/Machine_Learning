# Machine_Learning

> 一个面向系统化学习与实践经典机器学习算法的仓库，主要通过 Jupyter Notebook 展示特征工程、分类、回归、聚类与集成学习等内容，并配有完整可运行的 sklearn 示例与数据分析流程。  
> This repository is a practical notebook collection for learning classical machine learning: feature engineering, classification, regression, clustering, and ensemble learning, all implemented with scikit-learn and real datasets.

---

## 内容概览 (Overview)

当前仓库以交互式 Notebook 为主，配套少量 Python 脚本，涵盖如下核心主题：

- **特征工程 (Feature Engineering)**  
  - 字典特征向量化、文本特征提取（英文 / 中文）
  - 词频统计 (CountVectorizer) 与 TF–IDF (TfidfVectorizer)
  - 中文分词（基于 `jieba`）
  - 特征缩放（归一化 / 标准化）
  - 缺失值处理、低方差特征过滤
  - PCA 主成分分析降维与方差解释率分析

- **监督学习：分类 (Classification)**  
  - 鸢尾花 (Iris) 多分类问题完整流程
  - 文本分类：20 类新闻数据集 (20 Newsgroups)，TF-IDF + 朴素贝叶斯
  - K 近邻 (KNN) 在 Facebook Check-in 位置预测上的应用
  - 决策树、随机森林等分类器的基础使用
  - 数据集加载 `load_*` vs `fetch_*` 的区别与实践
  - ROC AUC、分类报告等评估指标

- **监督学习：回归 (Regression)**  
  - 加州房价数据集 (California Housing) 预测
  - 线性回归 (LinearRegression) 正规方程解
  - 随机梯度下降回归 (SGDRegressor) 及学习率、正则化调节
  - 岭回归 (Ridge) 与 L2 正则化
  - Lasso 回归 (Lasso) 与特征稀疏化
  - 均方误差 (MSE) 的概念与手工/库函数计算对比
  - 模型持久化与加载 (`joblib.dump` / `joblib.load`)

- **逻辑回归与二分类 (Logistic Regression & Binary Classification)**  
  - 对数损失函数可视化（`-log(x)`、`-log(1-x)` 曲线）
  - 乳腺癌数据集 (Wisconsin Breast Cancer) 良 / 恶性预测
  - 缺失值处理（`?` → `NaN` → 删除）、类型转换
  - 分类报告、ROC AUC 等评估指标的使用

- **无监督学习：聚类与用户行为画像 (Clustering)**  
  - Instacart 电商购物数据集多表关联与特征构造
    - 订单–商品–过道–用户 四表合并
    - 交叉表 (`pd.crosstab`) 构建「用户 × 商品过道」高维稀疏矩阵
  - KMeans 聚类用户购物偏好
  - PCA 用于高维用户特征降维与可视化
  - 轮廓系数 (silhouette score) 评估聚类质量
  - 大规模数据内存使用与性能考虑

- **集成学习 (Ensemble Learning)**  
  - 基于 `make_moons` 数据集的二分类实验
  - 基础单模型：
    - 逻辑回归 (`LogisticRegression`)
    - 支持向量机 (`SVC`)
    - 决策树 (`DecisionTreeClassifier`)
  - 手工投票集成：
    - 多模型预测结果相加并按“少数服从多数”进行投票
  - VotingClassifier 并行集成：
    - `voting='hard'` 与 `voting='soft'` 的对比与权重理解
  - Bagging 与随机森林：
    - `BaggingClassifier` + 决策树，`oob_score_` 利用袋外样本评估
    - `n_jobs=-1` 多核并行与时间对比
    - `bootstrap_features` / `max_features` 进行特征子采样
    - `RandomForestClassifier` 与 `ExtraTreesClassifier` 的使用与 oob 评估
  - Boosting：
    - AdaBoostClassifier（弱分类器叠加）
    - GradientBoostingClassifier (GBDT) 在二分类任务上的表现

---

## 文件结构 (Current Files)

当前仓库主要文件包括（示例）：

- `1-Feature Engineering.ipynb`  
  特征工程与预处理：
  - 字典特征提取：`DictVectorizer`
  - 英文文本向量化：`CountVectorizer`
  - 中文文本特征化：`CountVectorizer` + `jieba` 分词
  - TF–IDF：`TfidfVectorizer`
  - 数值特征缩放：`MinMaxScaler`, `StandardScaler`
  - 缺失值填补：`SimpleImputer`
  - 特征选择与降维：`VarianceThreshold`, `PCA`
  - 随机数与可视化练习

- `2-Classifier.ipynb`  
  分类任务与数据集加载：
  - Iris、20 Newsgroups、California Housing 的加载与探索
  - 训练 / 测试集划分 (`train_test_split`)
  - 特征标准化与 KNN、朴素贝叶斯、决策树、随机森林等分类器
  - Facebook Check-in 位置预测完整数据清洗与建模流程
  - 指标：`classification_report`, `roc_auc_score`

- `3-Regression Algorithm.ipynb`  
  回归与线性模型：
  - 基于加州房价数据的线性回归预测
  - 正规方程 vs 梯度下降 (SGDRegressor) 对比
  - Ridge / Lasso 回归与系数分析
  - 使用 `joblib` 进行模型保存与加载，模拟线上推理
  - MSE 的理论推导与实践对照
  - 逻辑回归基础、对数损失函数图像
  - 乳腺癌数据集上的逻辑回归二分类实战

- `4-Clustering.ipynb`  
  Instacart 用户行为聚类：
  - 多表 `merge` 建立用户–商品–过道关系
  - `pd.crosstab` 构造用户 × 过道频次矩阵
  - 缺失率分析、数据质量检查
  - PCA 降维 + KMeans 聚类 + 轮廓系数评估
  - 基于购买过道的用户兴趣画像示例

- `5-Ensemble Learning.ipynb`  
  集成学习与模型融合：
  - `make_moons` 合成数据集生成与可视化
  - 逻辑回归 / SVM / 决策树 单模型性能对比
  - 手动投票集成实现 & 准确率计算
  - VotingClassifier (hard / soft voting)
  - BaggingClassifier + 决策树，含：
    - oob 评估、`n_jobs` 对比、`max_samples` 参数实践
    - bootstrap + feature sampling 示例
  - 随机森林 / ExtraTrees 在同一任务上的效果与效率
  - AdaBoost & Gradient Boosting (GBDT) 串行集成示例

- `01.py`, `02.py`, `03.py`, `03_Data Saving.py`, `04.py`, `svm.ipynb`, `ovo_vs_ovr_benchmark..py` 等  
  - 部分为脚本化的算法练习或补充示例（如 SVM one-vs-one / one-vs-rest 对比基准等），可配合 Notebook 一同阅读。

---

## 环境与依赖 (Environment & Dependencies)

推荐使用 Python 3.x 与虚拟环境，常用依赖包括：

- **核心库**
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`（可选，用于更美观的可视化）
- **中文文本处理**
  - `jieba`
- **模型保存**
  - `joblib`

可根据 Notebook 中使用的模块，自行补充安装。

---

## 快速开始 (Quick Start)

```bash
# 克隆仓库
git clone https://github.com/logcoke/Machine_Learning.git
cd Machine_Learning

# 建议创建虚拟环境
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 安装常用依赖（示例）
pip install -U pip
pip install numpy pandas scikit-learn matplotlib seaborn jieba joblib

# 启动 Jupyter
pip install notebook  # 如未安装
jupyter notebook
```

打开对应的 `*.ipynb` 文件即可交互式运行、修改与实验。

---

## 学习路径建议 (Suggested Learning Path)

1. **特征工程与预处理**  
   从 `1-Feature Engineering.ipynb` 入手，理解特征表示、文本向量化、标准化、缺失值与降维等概念，为后续所有任务打基础。

2. **基础分类任务**  
   阅读 `2-Classifier.ipynb`：
   - 先从 Iris 数据集理解完整 ML 流程（探索 → 划分 → 训练 → 评估）
   - 再尝试 20 类新闻文本分类、KNN 地理位置预测，体会不同特征与模型的差异。

3. **回归与线性模型**  
   阅读 `3-Regression Algorithm.ipynb`：
   - 理解线性回归、梯度下降、正则化（Ridge / Lasso）
   - 学会模型保存/加载与指标（MSE）的含义

4. **逻辑回归与二分类**  
   在回归 Notebook 后半部分继续：
   - 熟悉对数损失、Sigmoid 与概率输出
   - 完成乳腺癌良 / 恶性分类任务，关注数据清洗（缺失值、异常值）

5. **聚类与推荐场景**  
   阅读 `4-Clustering.ipynb`：
   - 学会通过多表关联构造特征
   - 使用 PCA + KMeans 对用户进行聚类并分析簇特征

6. **集成学习与模型融合**  
   阅读 `5-Ensemble Learning.ipynb`：
   - 对比单模型与 Voting / Bagging / RandomForest / ExtraTrees / AdaBoost / GBDT
   - 体会样本采样、特征采样、弱分类器叠加带来的性能变化

---

## 实验与代码风格约定 (Experiment & Coding Conventions)

- Notebook 中尽量保证：
  - 从头到尾可以 **一次性运行通过**
  - 所有随机性均设置 `random_state` 以便复现
  - 对关键步骤（数据预处理 / 模型选择 / 评估指标）配有简短说明

- 建议代码风格：
  - 遵循 PEP 8
  - 可使用 `black` / `ruff` 等工具自动格式化脚本文件

---

## 适合人群 (Who Is This For)

- 希望用 **sklearn + Notebook** 快速上手机器学习的同学
- 已了解基础 Python / Numpy / Pandas，想系统整理传统机器学习知识的人
- 对特征工程、模型评估、集成学习等常见工程问题感兴趣的开发者

---

## 可能的后续扩展 (Possible Extensions)

未来可以考虑增加：

- 更系统的项目结构（`src/`, `datasets/`, `experiments/`, `docs/` 等）
- 深度学习入门篇（使用 PyTorch / TensorFlow 完成简单分类任务）
- 使用 `MLflow` 或自定义工具记录实验结果与参数
- 更多实际业务案例（如 CTR 预估、信用评分、异常检测等）

---

## 贡献指南 (Contributing)

欢迎通过 Issue / Pull Request 参与完善：

- 增补 / 修正文档与注释
- 新增 Notebook（例如：其它分类 / 回归 / 聚类案例）
- 优化现有代码结构、封装通用工具函数
- 增加更多可复现实验与可视化

---

Enjoy Learning & Experimenting!
