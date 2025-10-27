from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np
from sklearn.impute import SimpleImputer


# 特征抽取
#


def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    # sparse改为True,输出的是每个不为零位置的坐标，稀疏矩阵可以节省存储空间
    dict = DictVectorizer(sparse=False)  # 把sparse改为True看看

    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京', 'temperature': 100},
                               {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}])
    print(data)
    print('-' * 50)
    print(dict.get_feature_names())  # 字典中的一些类别数据，分别进行转换成特征
    print('-' * 50)
    print(dict.inverse_transform(data))

    return None


def couvec():
    # 实例化CountVectorizer
    # max_df, min_df整数：指每个词的所有文档词频数不小于最小值，出现该词的文档数目小于等于max_df
    # max_df, min_df小数：每个词的次数／所有文档数量

    vector = CountVectorizer()

    # 调用fit_transform输入并转换数据

    res = vector.fit_transform(
        ["life is  short,i like python life", "life is too long,i dislike python", "life is short"])

    # 打印结果,把每个词都分离了
    print(vector.get_feature_names())
    print(res)
    print(type(res))
    # 对照feature_names，标记每个词出现的次数
    print(res.toarray())


def countvec():
    """
    对文本进行特征值化,单个汉字单个字母不统计，因为单个汉字字母没有意义
    :return: None
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["人生苦短，我 喜欢 python python", "人生漫长，不用 python"])

    print(cv.get_feature_names())

    print(data)
    print(data.toarray())

    return None


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    print(type(con1))
    print('-' * 50)
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # print(content1, content2, content3)  #打印展示
    # 把列表转换成字符串
    print('-' * 50)
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def tfidfvec():
    """
    中文特征值化,倒排索引
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)
    print(type([c1,c2,c3]))
    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化处理
    :return: NOne
    """
    # 归一化容易受极值的影响
    mm = MinMaxScaler(feature_range=(0, 1))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)

    return None


def stand():
    """
    标准化缩放，不是标准正太分布，只均值为0，方差为1的分布
    :return:
    """
    std = StandardScaler()

    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)

    print(std.mean_)
    print(std.var_)
    print(std.n_samples_seen_)  # 样本数
    print('-' * 50)
    data1 = std.fit_transform([[-1.06904497, -1.35873244, 0.98058068],
                               [-0.26726124, 0.33968311, 0.39223227],
                               [1.33630621, 1.01904933, -1.37281295]])
    print(data1)
    # 均值
    print(std.mean_)
    # 方差
    print(std.var_)
    # 样本数
    print(std.n_samples_seen_)
    return None


def im():
    """
    缺失值处理
    :return:NOne
    """
    # NaN, nan,缺失值必须是这种形式，如果是？号，就要replace换成这种
    im = SimpleImputer(missing_values=np.nan, strategy='mean')

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)

    return None


def var():
    """
    特征选择-删除低方差的特征
    :return: None
    """
    var = VarianceThreshold()

    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)
    # 获得剩余的特征的列编号
    print('The surport is %s' % var.get_support(True))
    return None


def pca():
    """
    主成分分析进行特征降维
    :return: None
    """
    pca = PCA(n_components=1)

    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])

    print(data)

    return None


if __name__ == "__main__":
    # dictvec()
    # couvec()  # 针对数组中的1和0代表什么含义后面再看
    # countvec()
    # cutword()
    # hanzivec()
    # tfidfvec()
    # mm()
    stand()
    # im()
    # var()
    # pca()
