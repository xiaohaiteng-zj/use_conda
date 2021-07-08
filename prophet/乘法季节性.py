#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 10:58
# @Author  : 小海腾

import pandas as pd
from fbprophet import Prophet


df = pd.read_csv('examples/example_air_passengers.csv')
m = Prophet(seasonality_mode='multiplicative')
m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
# fig = m.plot(forecast).show()
m.plot_components(forecast).show()
