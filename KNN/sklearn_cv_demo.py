#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 22:50
# @Author  : 小海腾

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
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
knn = KNeighborsClassifier()

# 4.1、模型调优--交叉验证，网格搜索
param_grid = {"n_neighbors": [1, 3, 7]}
knn = GridSearchCV(knn, param_grid=param_grid, cv=5)

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

# 5.3、查看交叉验证，网格搜索的属性
print("在交叉验证中得到的最好的结果是：\n", knn.best_score_)
print("在交叉验证中得到的最好的模型是：\n", knn.best_estimator_)
print("在交叉验证中得到的最好的参数模型是：\n", knn.best_params_)
print("在交叉验证中得到的模型结果是：\n", knn.cv_results_)
