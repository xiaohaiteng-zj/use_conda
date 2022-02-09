#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/27 16:50
# @Author  : 小海腾
from sklearn.linear_model import LinearRegression

# 1、获取数据
x = [[80, 86], [82, 80], [85, 78], [90, 90],
     [86, 82], [82, 90], [78, 80], [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

# 2、模型训练
# 2.1、实例化一个估计器对象
estimator = LinearRegression()
# 2.2、调用fit方法，进行训练
estimator.fit(x, y)

# 3、数据预测
re = estimator.predict([[90, 80]])
print(re)

# 4、打印对应的系数
print("线性回归的系数是：\n", estimator.coef_)
