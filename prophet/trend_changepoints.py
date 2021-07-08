#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 10:40
# @Author  : 小海腾


import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

df = pd.read_csv('examples/example_wp_log_peyton_manning.csv')

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
a.show()
