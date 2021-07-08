#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 14:33
# @Author  : 小海腾


import pandas as pd
from fbprophet import Prophet


df = pd.read_csv('examples/example_retail_sales.csv')
m = Prophet().fit(df)
# future = m.make_future_dataframe(periods=3652)
# fcst = m.predict(future)
# m.plot(fcst).show()

future = m.make_future_dataframe(periods=120, freq='M')
fcst = m.predict(future)
m.plot(fcst).show()
