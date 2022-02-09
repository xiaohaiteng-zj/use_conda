#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 15:13
# @Author  : 小海腾

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

knn = KNeighborsClassifier()

param_grid = {"n_neighbors": [3, 5, 7, 9]}
knn = GridSearchCV(knn, param_grid=param_grid, cv=5)

knn.fit(x_train, y_train)

y_pre = knn.predict(x_test)
print(y_pre)
print(y_test)
print(y_pre == y_test)

score = knn.score(x_test, y_test)
print(score)

print(knn.best_params_)