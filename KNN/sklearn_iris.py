#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 17:08
# @Author  : 小海腾

# 鸢尾花案例
from sklearn.datasets import load_iris, fetch_20newsgroups

# 小数据集
iris = load_iris()
# print(iris)

# 大数据集
# news = fetch_20newsgroups()
# print(news)

# 数据集属性描述
print("数据集的特征值是：\n", iris.data)
print("数据集的目标值是：\n", iris.target)
print("数据集的特征值名字是：\n", iris.feature_names)
print("数据集的目标值名字是：\n", iris.target_names)
print("数据集的描述：\n", iris.DESCR)
