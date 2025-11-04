from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


# sklearn 必须是np
def base_datas():
    # 先看分类的数据
    li = load_iris()
    #
    # print("获取特征值")
    # print(type(li.data))
    # print(li.data)
    # print(li.data.shape)
    # print("目标值")
    # print(li.target)
    # print('-' * 50)
    print(li.DESCR)
    # print('-' * 50)
    # print(li.feature_names)  # 重点
    # print('-' * 50)
    # print(li.target_names)
    # print('-' * 50)
    # # # 注意返回值, 训练集 train  x_train, y_train        测试集  test   x_test, y_test，顺序千万别搞错了
    # # # 默认是乱序的,random_state为了确保两次的随机策略一致
    # x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25, random_state=1)
    # #
    # print("训练集特征值和目标值：", x_train, y_train)
    # print("测试集特征值和目标值：", x_test, y_test)

    # 下面是比较大的数据，需要下载一会
    # news = fetch_20newsgroups(subset='all', data_home='data')
    # # print(news.feature_names)  #这个没有
    # print(news.DESCR)
    # print('-' * 50)
    # print(news.data[0])
    # print(len(news.data))
    # print('-' * 50)
    # print(news.target)
    # print(min(news.target), max(news.target))
    # 接着来看回归的数据
    lb = load_boston()

    print("获取特征值")
    print(lb.data[0])
    print(lb.data.shape)
    print("目标值")
    print(lb.target)
    print(lb.DESCR)
    print(lb.feature_names)
    print('-' * 50)
    # 回归问题没这个
    # print(lb.target_names)


def knncls():
    """
    K-近邻预测用户签到位置
    :return:None
    """
    # 读取数据
    data = pd.read_csv("./data/FBlocation/train.csv")

    print(data.head(10))

    # 处理数据
    # 1、缩小数据,查询数据
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")

    # 处理时间的数据
    time_value = pd.to_datetime(data['time'], unit='s')

    print(time_value)
    #
    # 把日期格式转换成 字典格式，把年，月，日，时，分，秒转换为字典格式
    time_value = pd.DatetimeIndex(time_value)
    #
    print('-' * 50)
    print(time_value)
    print('-' * 50)
    # 构造一些特征，执行的警告是因为我们的操作是复制，loc是直接放入
    print(type(data))
    # data['day'] = time_value.day
    # data['hour'] = time_value.hour
    # data['weekday'] = time_value.weekday
    data.insert(data.shape[1], 'day', time_value.day)
    data.insert(data.shape[1], 'hour', time_value.hour)
    data.insert(data.shape[1], 'weekday', time_value.weekday)

    #
    # 把时间戳特征删除
    data = data.drop(['time'], axis=1)

    print(data)
    #
    # # 把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()
    # # 把index变为0,1,2，3,4,5,6这种效果，从零开始拍，原来的index是地点
    tf = place_count[place_count.row_id > 3].reset_index()
    # 根据设定的地点目标值，对原本的样本进行过滤
    data = data[data['place_id'].isin(tf.place_id)]
    print(data)
    # # 取出数据当中的特征值和目标值
    y = data['place_id']
    # 删除目标值，保留特征值，
    x = data.drop(['place_id'], axis=1)
    # 删除无用的特征值
    x = x.drop(['row_id'], axis=1)
    # 进行数据的分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    # 特征工程（标准化）,下面3行注释，一开始我们不进行标准化，看下效果，目标值要不要标准化？
    std = StandardScaler()
    # #
    # # # 对测试集和训练集的特征值进行标准化,服务于knn fit
    # x_train = std.fit_transform(x_train)
    # # transform返回的是copy，不在原有的输入对象中去修改
    # print(id(x_test))
    # x_test = std.transform(x_test)
    # print(id(x_test))
    #
    # # 进行算法流程 # n_neighbors是超参数，可以通过设置n_neighbors=5，来调整结果好坏
    knn = KNeighborsClassifier(n_neighbors=5)

    # # fit， predict,score，训练
    # knn.fit(x_train, y_train)
    # # #
    # # # 得出预测结果
    # y_predict = knn.predict(x_test)
    # #
    # print("预测的目标签到位置为：", y_predict)
    # # #
    # # # # 得出准确率
    # print("预测的准确率:", knn.score(x_test, y_test))

    # 下面的是超参数搜索-网格搜索API部分演示
    # # 构造一些参数的值进行搜索
    param = {"n_neighbors": [3, 5, 10, 12, 15]}
    #
    # 进行网格搜索，cv=3是3折交叉验证，用其中2折训练，1折验证
    gc = GridSearchCV(knn, param_grid=param, cv=3)
    # 这里应该传x,y
    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    gc.fit(x_train, y_train)
    x_test = std.transform(x_test)
    # 预测准确率，为了给大家看看
    print("在测试集上准确率：", gc.score(x_test, y_test))

    print("在交叉验证当中最好的结果：", gc.best_score_)

    print("选择最好的模型是：", gc.best_estimator_)

    print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    return None


def naviebayes():
    """
    朴素贝叶斯进行文本分类
    :return: None
    """
    news = fetch_20newsgroups(subset='all', data_home='data')

    print(len(news.data))
    print(news.target)
    print(news.target_names)
    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=1)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)

    # print(tf.get_feature_names())

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测,alpha是拉普拉斯平滑系数
    mlt = MultinomialNB(alpha=1.0)

    print(x_train.toarray())
    # 训练
    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)

    print("预测的文章类别为：", y_predict)

    # 得出准确率,这个是很难提高准确率，为什么呢？
    print("准确率为：", mlt.score(x_test, y_test))
    # 目前这个场景我们不需要召回率，support是划分为那个类别的有多少个样本
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))
    # 把0-19总计20个分类，变为0和1
    y_test = np.where(y_test == 5, 1, 0)
    y_predict = np.where(y_predict == 5, 1, 0)
    # roc_auc_score的y_test只能是二分类
    print("AUC指标：", roc_auc_score(y_test, y_predict))
    return None


def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据
    titan = pd.read_csv("./data/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]

    y = titan['survived']
    x.info()  # 用来判断是否有空值
    print(x)
    # 一定要进行缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)
    print(x_train.head())
    # 进行处理（特征工程）特征-》类别-》one_hot编码
    dict = DictVectorizer(sparse=False)

    # 这一步是对字典进行特征抽取
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(type(x_train))
    print(dict.get_feature_names())
    print('-' * 50)
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # print(x_train)
    # # 用决策树进行预测，修改max_depth试试
    # dec = DecisionTreeClassifier()
    #
    # dec.fit(x_train, y_train)
    # #
    # # # 预测准确率
    # print("预测的准确率：", dec.score(x_test, y_test))
    # #
    # # # 导出决策树的结构
    # export_graphviz(dec, out_file="tree.dot",
    #                 feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 随机森林进行预测 （超参数调优）
    rf = RandomForestClassifier(n_jobs=-1)
    # 120, 200, 300, 500, 800, 1200
    param = {"n_estimators": [50, 80], "max_depth": [2, 3, 5, 8, 15, 25]}

    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=3)

    gc.fit(x_train, y_train)

    print("准确率：", gc.score(x_test, y_test))

    print("查看选择的参数模型：", gc.best_params_)

    print("选择最好的模型是：", gc.best_estimator_)

    print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    return None


if __name__ == "__main__":
    # base_datas()
    # knncls()
    naviebayes()
    # decision()
