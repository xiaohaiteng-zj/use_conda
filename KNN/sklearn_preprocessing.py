#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/26 22:24
# @Author  : 小海腾
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minmax_demo():
    """
    归一化演示
    :return:
    """
    data = pd.DataFrame({"里程": [40920, 14488, 26052, 75136],
                         "公升": [8.32, 7.15, 1.44, 13.14],
                         "时间占比": [0.95, 1.67, 0.80, 0.41],
                         "类别": [3, 2, 1, 1]})
    print(data)

    # 1、实例化
    transfer = MinMaxScaler(feature_range=(3, 5))
    # 2、进行转化
    re = transfer.fit_transform(data[["里程", "公升", "时间占比"]])
    print(re)


def stand_demo():
    """
    标准化演示
    :return:
    """
    data = pd.DataFrame({"里程": [40920, 14488, 26052, 75136],
                         "公升": [8.32, 7.15, 1.44, 13.14],
                         "时间占比": [0.95, 1.67, 0.80, 0.41],
                         "类别": [3, 2, 1, 1]})
    print(data)

    # 1、实例化
    transfer = StandardScaler()
    # 2、进行转化
    re = transfer.fit_transform(data[["里程", "公升", "时间占比"]])
    print(re)


# minmax_demo()
stand_demo()