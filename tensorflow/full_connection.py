#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 10:42
# @Author  : 小海腾
import tensorflow as tf


def full_connection():
    """
    用全连接来对手写数字进行识别
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    # 1、获取数据
    mint = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mint.load_data()
    print(x_train.shape)
    print(y_train)
    print(x_test.shape)
    print(y_test)
    x = tf.reshape(x_train, shape=(-1, 28, 28, 1))
    print(x)
    print(x.shape)
    # x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 784))
    # y_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10))
    x = tf.cast(x_train, tf.float32)
    y_true = tf.cast(y_train, tf.float32)

    # 2、构建模型
    # weights = tf.Variable(initial_value=tf.random.normal(shape=(784, 10)))
    # bias = tf.Variable(initial_value=tf.random.normal(shape=[10]))
    # y_predict = tf.matmul(x, weights) + bias

    # # 3、构造损失函数】
    # error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # # 4、优化损失函数
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    #
    # # 初始化变量
    # init = tf.compat.v1.global_variables_initializer()
    # with tf.compat.v1.Session() as sess:
    #     # 运行初始化
    #     sess.run(init)
    #     print("训练前模型参数：损失%f" % (error.eval()))
    #
    #     # 开始训练
    #     for i in range(1000):
    #         sess.run(optimizer)
    #
    #         print("训练后模型参数：损失%f" % (error.eval()))


if __name__ == '__main__':
    full_connection()
