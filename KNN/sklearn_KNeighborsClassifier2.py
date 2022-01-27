#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 15:11
# @Author  : 小海腾

from sklearn.neighbors import KNeighborsClassifier

# 机器学习中距离度量的方法
# 1、欧氏距离
# 2、曼哈顿距离
# 3、切比雪夫距离
# 4、闵氏距离
# 5、标准欧式距离（前四个都把单位相同看待了，所以不太科学，解决方法）

# 1、准备数据
x = [[39, 0, 31],
     [3, 2, 65],
     [2, 3, 55],
     [9, 38, 2],
     [8, 34, 17],
     [5, 2, 57],
     [21, 17, 5],
     [45, 2, 9]
     ]
y = ["喜剧片", "动作片", "爱情片", "爱情片", "爱情片", "动作片", "喜剧片", "喜剧片"]

# 2、训练模型
# 2.1、实例化一个估计器对象
estimator = KNeighborsClassifier(n_neighbors=5)
# 2.2、调用fit方法，进行训练
estimator.fit(x, y)

# 3、数据预测
re = estimator.predict([[23, 3, 17]])
print(re)
