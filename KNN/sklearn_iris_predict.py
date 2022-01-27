#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 22:50
# @Author  : 小海腾

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1、获取数据
iris = load_iris()

# 2、数据基本处理
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 3、特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4、训练模型-KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# 5、模型评估
# 5.1、预测值结果输出
y_pre = knn.predict(x_test)
print("预测值是：\n", y_pre)
print("真实值是：\n", y_test)
print("对比值是：\n", y_pre == y_test)

# 5.2、准确率计算
score = knn.score(x_test, y_test)
print("准确率为：\n", score)
