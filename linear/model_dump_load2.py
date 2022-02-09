#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 22:59
# @Author  : 小海腾
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import joblib

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=22)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
estimator.fit(x_train, y_train)

joblib.dump(estimator, "./data/test2.pkl")

print("模型的偏置是：\n", estimator.intercept_)
print("模型的系数是：\n", estimator.coef_)

y_pre = estimator.predict(x_test)
ret = mean_squared_error(y_test, y_pre)
print("均方误差是：\n", ret)
