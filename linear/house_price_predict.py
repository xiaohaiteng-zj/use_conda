#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 17:00
# @Author  : 小海腾
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


def linear_model1():
    """
    线性回归：正规方程
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
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)

    # 5、模型评估
    # 5.1、预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)

    # 5.2、均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差是：\n", ret)


def linear_model2():
    """
    线性回归：梯度下降
    :return:
    """
    # 1、获取数据
    boston = load_boston()

    # 2、数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    # 3、特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4、机器学习，模型训练（线性回归）
    # estimator = SGDRegressor(learning_rate="constant", eta0=0.001)
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)

    # 5、模型评估
    # 5.1、预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)

    # 5.2、均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差是：\n", ret)


def linear_model3():
    """
    线性回归：岭回归
    :return:
    """
    # 1、获取数据
    boston = load_boston()

    # 2、数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    # 3、特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4、机器学习，模型训练（线性回归）
    # alpha：正则化力度；正则化力度越大，权重系数越小；正则化力度越小，权重系数越大。
    # estimator = Ridge(alpha=1.0)
    estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)

    # 5、模型评估
    # 5.1、预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)

    # 5.2、均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差是：\n", ret)


# linear_model1()
# linear_model2()
linear_model3()
