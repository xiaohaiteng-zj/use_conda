#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 10:40
# @Author  : 小海腾


import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_yearly


df = pd.read_csv('examples/example_wp_log_peyton_manning.csv')
m = Prophet().fit(df)
future = m.make_future_dataframe(periods=365)

m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast).show()


