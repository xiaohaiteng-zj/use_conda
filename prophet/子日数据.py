#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 13:58
# @Author  : 小海腾

import pandas as pd
from fbprophet import Prophet


df = pd.read_csv('examples/example_yosemite_temps.csv')
m = Prophet(changepoint_prior_scale=0.01).fit(df)
future = m.make_future_dataframe(periods=300, freq='H')
fcst = m.predict(future)
# fig = m.plot(fcst).show()

fig = m.plot_components(fcst).show()
