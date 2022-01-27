#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 15:11
# @Author  : 小海腾

from sklearn.neighbors import KNeighborsClassifier

# 1、准备数据
x = [[1], [2], [10], [20]]
y = [0, 1, 2, 3]

# 2、训练模型
# 2.1、实例化一个估计器对象
estimator = KNeighborsClassifier(n_neighbors=2)
# 2.2、调用fit方法，进行训练
estimator.fit(x, y)

# 3、数据预测
re = estimator.predict([[0]])
print(re)

re = estimator.predict([[11]])
print(re)
