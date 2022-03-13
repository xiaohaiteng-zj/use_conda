#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/6 22:58
# @Author  : 小海腾

import tensorflow as tf
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# 1.准备数据
x = tf.random.normal(shape=[100, 1])
y_true = tf.matmul(x, [[0.8]]) + 0.7

# 4.优化损失
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')
model.fit(x, y_true, epochs=50)
re = model.predict(pd.Series([10]))
print(re)

# print("训练前模型参数：权重%f, 偏置%f, 损失%f" % (weights.eval(), bias.eval(), error.eval()))
# # 开始训练
# # for i in range(1000):
# #     optimizer
# print("训练后模型参数：权重%f, 偏置%f, 损失%f" % (weights.eval(), bias.eval(), error.eval()))


