from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
import joblib
import pandas as pd
import numpy as np


def mylinear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()

    print("获取特征值")
    print(lb.data)
    print("目标值")
    print(lb.target)
    print(lb.DESCR)
    print(lb.feature_names)
    print('-' * 50)
    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25, random_state=1)
    #
    print(y_test)
    #
    # # 进行标准化处理(?) 目标值处理？
    # # 特征值和目标值是都必须进行标准化处理, 实例化两个标准化API
    std_x = StandardScaler()
    #
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    #
    # # 目标值进行了标准化
    # std_y = StandardScaler()
    # #
    # temp = y_train.reshape(-1, 1)
    # y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 目标值是一维的，这里需要传进去2维的
    # y_test = std_y.transform(y_test.reshape(-1, 1))
    print('-' * 50)
    # print(y_train)
    # # 预测房价结果，掌握如何加载训练好的模型
    # model = joblib.load("./tmp/test.pkl")
    # # # 因为目标值进行了标准化，一定要把预测后的值逆向转换回来
    # y_predict = model.predict(x_test)
    # y_predict = std_y.inverse_transform(model.predict(x_test))
    # #
    # print("保存的模型预测的结果：", y_predict)
    # print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))
    # ----------------------------------------------
    # # estimator预测
    # # # 正规方程求解方式预测结果，正规方程进行线性回归
    lr = LinearRegression()
    # #
    lr.fit(x_train, y_train)
    #
    print('回归系数', lr.coef_)  # 回归系数可以看特征与目标之间的相关性
    #
    y_predict = lr.predict(x_test)
    #
    # print(y_predict)  # 这里会有负的，因为这是标准化之后的
    #
    # # # 预测测试集的房子价格，通过inverse得到真正的房子价格
    # y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    # print("正规方程测试集里面每个房子的预测价格：", y_lr_predict)
    #
    # # 保存训练好的模型
    # joblib.dump(lr, "./tmp/test.pkl")
    #
    # print(y_lr_predict)
    # print('*' * 50)
    # print(std_y.inverse_transform(y_test))
    # print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    print("正规方程的均方误差：", mean_squared_error(y_test, y_predict))
    # # 梯度下降去进行房价预测,数据量大要用这个
    # 默认可以去调 eta0 = 0.008，会改变learning_rate
    # learning_rate='optimal',alpha会影响学习率的值，由alpha来算学习率
    # sgd = SGDRegressor(eta0=0.008)
    # # # 训练
    # sgd.fit(x_train, y_train)
    # #
    # # print('梯度下降的回归系数', sgd.coef_)
    # #
    # # # 预测测试集的房子价格
    # # y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    # y_predict = sgd.predict(x_test)
    # # print("梯度下降测试集里面每个房子的预测价格：", y_sgd_predict)
    # print("梯度下降的均方误差：", mean_squared_error(y_test, y_predict))
    # print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
    # # #
    # # # # 岭回归去进行房价预测
    # rd = Ridge(alpha=0.2)
    #
    # rd.fit(x_train, y_train)
    #
    # print(rd.coef_)
    # #
    # # # 预测测试集的房子价格
    # # y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    # y_predict = rd.predict(x_test)
    # # print("岭回归里面每个房子的预测价格：", y_rd_predict)
    # #
    # print("岭回归的均方误差：", mean_squared_error(y_test, y_predict))
    # print("岭回归的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))
    #
    # return None


def logistic():
    """
    逻辑回归做二分类进行癌症预测（根据细胞的属性特征）
    :return: NOne
    """
    # 构造列标签名字
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']

    # 读取数据
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column)

    print(data)

    # 缺失值进行处理
    data = data.replace(to_replace='?', value=np.nan)
    # 直接删除
    data = data.dropna()
    print('-' * 50)
    print(data)
    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25,
                                                        random_state=1)

    # 进行标准化处理
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    #
    # # 逻辑回归预测
    # C正则化力度
    # solver = 'liblinear'
    lg = LogisticRegression(C=1)
    #
    lg.fit(x_train, y_train)
    # 逻辑回归的权重参数
    print(lg.coef_)

    y_predict = lg.predict(x_test)
    print(y_predict)
    print("准确率：", lg.score(x_test, y_test))
    print(lg.predict_proba(x_test))
    # 为什么还要看下召回率，labels和target_names对应
    # macro avg 平均值  weighted avg 加权平均值
    print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))

    print("AUC指标：", roc_auc_score(y_test, y_predict))
    return None


if __name__ == "__main__":
    # 线性回归
    # mylinear()
    logistic()
