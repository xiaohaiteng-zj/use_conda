#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 17:50
# @Author  : 小海腾
import time
import random

import numpy as np

a = []
for i in range(100000000):
    a.append(random.random())

print("----------")

strat = time.time()
sum1 = sum(a)
end = time.time()
print("计算时间", end - strat)

b = np.array(a)
strat = time.time()
sum2 = np.sum(b)
end = time.time()
print("计算时间", end - strat)
