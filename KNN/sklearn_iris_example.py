#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 23:22
# @Author  : 小海腾
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pre = knn.predict(x_test)
print(y_pre == y_test)

score = knn.score(x_test, y_test)
print(score)