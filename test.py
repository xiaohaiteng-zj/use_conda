#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/3 16:56
# @Author  : 小海腾

import pandas as pd
import numpy as np
import math
import random

import matplotlib.pyplot as plt


# x = [i for i in range(1, 25)]
# y_sh = [0, 0, 0, 0, 0, 0,
#         0, 0, 151, 134, 116, 116,
#         105, 121, 104, 103, 93, 116,
#         138, 103, 100, 0, 0, 0]
#
# plt.figure(figsize=(20, 8), dpi=100)
# plt.plot(x, y_sh)
# plt.show()

lis1 = [460, 225, 119, 82, 111, 269,
        613, 761, 605, 555, 638, 684,
        902, 837, 576, 450, 639, 892,
        1111, 1434, 1729, 1718, 1403, 995]

lis2 = [0, 0, 0, 0, 0, 0,
        0, 0, 126, 111, 97, 96,
        87, 101, 87, 86, 77, 97,
        114, 86, 83, 0, 0, 0]


df_day = pd.DataFrame({
    "order": [i for i in range(1, 25)],
    "trend": lis2
})


for i in range(8, 18):
    avg = math.ceil((lis2[i] + lis2[i+1] + lis2[i+2] + lis2[i+3]) / 4)
    lis2[i] = avg
    lis2[i+1] = avg
    lis2[i+2] = avg
    lis2[i+3] = avg

re = lis2
print(re)

x = [i for i in range(1, 25)]
y_sh = [0, 0, 0, 0, 0, 0,
        0, 0, 126, 111, 97, 96,
        87, 101, 87, 86, 77, 97,
        114, 86, 83, 0, 0, 0]

y_sh2 = [0, 0, 0, 0, 0, 0,
         0, 0, 108, 103, 103, 99,
         96, 92, 94, 99, 96, 93,
         93, 93, 93, 0, 0, 0]

plt.figure(figsize=(20, 8), dpi=100)
plt.plot(x, y_sh)
plt.plot(x, y_sh2)
plt.show()


# y_sh2 = re
# plt.figure(figsize=(20, 8), dpi=100)
# plt.plot(x, y_sh2)
# plt.show()

print("hahahahahhah")
