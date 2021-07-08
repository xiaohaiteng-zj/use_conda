#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 17:57
# @Author  : 小海腾


import pandas as pd
from fbprophet import Prophet


df = pd.read_csv('examples/example_wp_log_R.csv')

# 预测饱和增长
# df['cap'] = 8.5
# print(df.tail())
#
# m = Prophet(growth='logistic')
# m.fit(df)
#
# future = m.make_future_dataframe(periods=1826)
# future['cap'] = 8.5
# fcst = m.predict(future)
# m.plot(fcst).show()

# ------------------------------------------------------------------------------------------

# 预测饱和减少
df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5

m = Prophet(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 6
future['floor'] = 1.5
fcst = m.predict(future)
m.plot(fcst).show()


