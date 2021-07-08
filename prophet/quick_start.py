#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 14:35
# @Author  : 小海腾

import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('examples/example_wp_log_peyton_manning.csv')
print(df.head())


# 拟合模型
m = Prophet()
m.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365)
print(future.tail())

# 预测数据集
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# 展示预测结果
m.plot(forecast).show()

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
m.plot_components(forecast).show()
