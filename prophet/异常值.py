#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 11:20
# @Author  : 小海腾


import pandas as pd
from fbprophet import Prophet

# df = pd.read_csv('examples/example_wp_log_R_outliers1.csv')
# m = Prophet()
# m.fit(df)
# future = m.make_future_dataframe(periods=1096)
# forecast = m.predict(future)
# m.plot(forecast).show()


# 将2010年一年的数据设为缺失
# df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
# model = Prophet().fit(df)
# model.plot(model.predict(future)).show()


df = pd.read_csv('examples/example_wp_log_R_outliers2.csv')
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=1096)
forecast = m.predict(future)
# m.plot(forecast).show()

# 将2015年前半年的数据设为缺失
df.loc[(df['ds'] > '2015-06-01') & (df['ds'] < '2015-06-30'), 'y'] = None
m = Prophet().fit(df)
m.plot(m.predict(future)).show()


