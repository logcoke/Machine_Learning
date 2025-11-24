from sklearn import datasets
from sklearn.linear_model import LogisticRegression

d = datasets.load_iris()
x = d.data
y = d.target
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# 默认的OVR的多分类任务，时间更短，准确度较低
log1 = LogisticRegression()
log1.fit(x_train, y_train)
print(log1.score(x_test, y_test))
# 修改默认参数，使得其成为OVO的多分类算法，准确度更高一点，时间更长
log2 = LogisticRegression(multi_class="multinomial", solver="newton-cg")
log2.fit(x_train, y_train)
print(log2.score(x_test, y_test))
# sklearn中封装的OVO和OVR
# sklearn中对于所有的二分类算法提供了统一的OVR和OVO的分类器函数，可以方便调用实现所有二分类算法的多分类实现
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

log_reg = LogisticRegression()  # 1-1定义一种二分类算法
ovr = OneVsRestClassifier(log_reg)  # 1-2进行多分类转换OVR
ovo = OneVsOneClassifier(log_reg)  # 1-2进行多分类转换OVO
ovr.fit(x_train, y_train)  # 1-3进行数据训练与预测
print(ovr.score(x_test, y_test))
ovo.fit(x_train, y_train)
print(ovo.score(x_test, y_test))
