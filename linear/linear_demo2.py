#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 16:09
# @Author  : 小海腾
from sklearn.linear_model import LinearRegression

# 1、获取数据
x = [[80, 86], [82, 80], [85, 78], [90, 90],
     [86, 82], [82, 90], [78, 80], [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

estimator = LinearRegression()
estimator.fit(x, y)

re = estimator.predict([[90, 80]])
print(re)

print(estimator.coef_)