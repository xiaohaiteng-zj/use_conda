

import pandas as pd
import numpy as np


df_index = pd.date_range('2018-11-11', '2021-10-01', freq='D')
dd = pd.DataFrame({'ds': df_index})
dd["week_day"] = dd['ds'].dt.weekday
dd["in_month"] = dd['ds'].dt.strftime('%d')
dd["in_quarter"] = dd['ds'].dt.dayofyear
dd["in_year"] = dd['ds'].dt.month

print(dd)
print(dd["in_quarter"][0])
print(type(dd["in_quarter"][0]))

df_week = pd.DataFrame({
    "week_order": [1, 2, 3, 4, 5, 6, 7],
    "trend": [0.07, -0.04, -0.03, -0.03, 0.09, -0.03, -0.03]
})

df_month = pd.DataFrame({
    "week_order": [i for i in range(1, 32)],
    "trend": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
})

df_quarter = pd.DataFrame({
    "week_order": [i for i in range(1, 92)],
    "trend": [0.8, 0.2, 0.3, 0.4, 0.5, 0.6,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
})

df_activity = pd.DataFrame({
    "date_order": pd.date_range('2021-06-03', '2021-10-02', freq='D'),
    "trend": [0 for i in range(122)]
})

df_year = pd.DataFrame({
    "week_order": [1, 2, 3, 4, 5, 6, 7, 8],
    "trend": [0.07, -0.04, -0.03, -0.03, 0.09, -0.03, -0.03]
})

print(df_activity)
df_activity.loc[df_activity["date_order"] == "2021-09-25", "trend"] = 0.1
df_activity.loc[df_activity["date_order"] == "2021-10-02", "trend"] = 0.9
print(df_activity.tail(10))
df_activity = df_activity.rename(columns={'trend': "activity_fact"})
# print(df_week.at[0, 'trend'])

df_week_list = df_week["trend"].to_list()
df_month_list = df_month["trend"].to_list()
df_quarter_list = df_quarter["trend"].to_list()

print(df_week_list)
print(df_month_list)
print(df_quarter_list)

dd['week_fact'] = dd["week_day"].apply(lambda x: df_week_list[x])
dd['month_fact'] = dd["in_month"].apply(lambda x: df_month_list[int(x)-1])
dd['quarter_fact'] = dd["in_quarter"].apply(lambda x: df_quarter_list[(x % 91)-1])

# for data in df_activity['date_order']:
# result = pd.concat([dd, df_activity], axis=1, join_axes=[dd.ds])
result = pd.merge(dd, df_activity, how='left', left_on=dd['ds'], right_on=df_activity['date_order'])

print(dd.tail(10))
print(result.tail(10))
print(result.fillna(value=0))


