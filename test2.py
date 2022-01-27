#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 17:50
# @Author  : 小海腾
import time
import random

import numpy as np
import pandas as pd


score = np.random.randint(40, 100, (10, 5))
cols = ["语文", "数学", "英语", "化学", "物理"]
df = pd.DataFrame(score, columns=cols)
print(df)

df[df["语文"] < 60] = np.nan

print(df)

a = np.any(pd.isnull(df))
print(a)

df.fillna(99, inplace=True)
print(df)
