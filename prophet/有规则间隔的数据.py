#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 14:06
# @Author  : 小海腾

import pandas as pd
from fbprophet import Prophet


df = pd.read_csv('examples/example_yosemite_temps.csv')

df2 = df.copy()
df2['ds'] = pd.to_datetime(df2['ds'])
# 只保留每天早上6点之前的数据
df2 = df2[df2['ds'].dt.hour < 6]
m = Prophet().fit(df2)
future = m.make_future_dataframe(periods=300, freq='H')
# fcst = m.predict(future)
# fig = m.plot(fcst).show()

future2 = future.copy()
future2 = future2[future2['ds'].dt.hour < 6]
fcst = m.predict(future2)
fig = m.plot(fcst).show()
