#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 17:25
# @Author  : 小海腾

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from pylab import mpl
# 设置显示中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 设置正确显示符号
mpl.rcParams['axes.unicode_minus'] = False


iris = load_iris()
# print(iris)

# 数据集属性描述
# print("数据集的特征值是：\n", iris.data)
# print("数据集的目标值是：\n", iris.target)
# print("数据集的特征值名字是：\n", iris.feature_names)
# print("数据集的目标值名字是：\n", iris.target_names)
# print("数据集的描述：\n", iris.DESCR)

# 数据可视化
iris_df = pd.DataFrame(data=iris.data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
iris_df["target"] = iris.target
print(iris_df)


def iris_plot(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue='target', fit_reg=False)
    plt.title("鸢尾花数据显示")
    plt.show()


# iris_plot(iris_df, 'sepal width (cm)', 'petal length (cm)')
# iris_plot(iris_df, 'sepal length (cm)', 'petal width (cm)')

# 数据集的划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print("训练集的特征值是：\n", x_train)
print("训练集的目标值是：\n", y_train)
print("测试集的特征值是：\n", x_test)
print("测试集的目标值是：\n", y_test)
