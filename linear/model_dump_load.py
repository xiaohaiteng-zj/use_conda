#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 18:23
# @Author  : 小海腾
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import joblib


def dump_load():
    """
    模型保存与加载
    :return:
    """
    # 1、获取数据
    boston = load_boston()

    # 2、数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=22)

    # 3、特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4、机器学习，模型训练（线性回归）
    # 4.1、模型训练
    estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)
    # 4.2、模型保存
    joblib.dump(estimator, "./data/test.pkl")
    # # 4.3、模型加载
    # estimator = joblib.load("./data/test.pkl")

    # 5、模型评估
    # 5.1、预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)

    # 5.2、均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差是：\n", ret)


dump_load()
