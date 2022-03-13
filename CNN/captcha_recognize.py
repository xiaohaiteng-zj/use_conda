#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/12 22:33
# @Author  : 小海腾

import tensorflow as tf
import os
import numpy as np


def read_picture():
    """
    读取图片数据
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    # 1、构造文件名队列
    # 构造路径+文件名列表
    filenames = os.listdir("./image")
    # 拼接路径+文件名
    file_list = [os.path.join("./image/", file) for file in filenames]
    file_queue = tf.compat.v1.train.string_input_producer(file_list)

    # 2、读取与解码
    reader = tf.compat.v1.WholeFileReader()
    # key是文件名，value是一个样本，也就是一张图片
    filename, image = reader.read(file_queue)

    # 解码阶段
    decoded = tf.image.decode_jpeg(image)

    # 图像的形状类型修改(生成的图片是60*160*3，为了和案例一致，进行缩放？是否有必要)
    image_resized = tf.compat.v1.image.resize_images(decoded, [20, 80])
    image_resized.set_shape([20, 80, 3])

    # 3、批处理
    filename_batch, image_batch = tf.compat.v1.train.batch([filename, image_resized], batch_size=100, num_threads=1, capacity=200)

    return filename_batch, image_batch


def filename2lable(filename):
    """
    将一个样本的特征值和目标值一一对应
    :param filename:
    :return:
    """
    lables = []
    for file_name in filename:
        real_name = str(file_name)[10:14]
        letter = []
        for word in real_name:
            letter.append(ord(word) - ord("A"))
        lables.append(letter)
    return np.array(lables)


def create_weights(shape):
    return tf.Variable(initial_value=tf.random.normal(shape=shape, stddev=0.01))


def creat_model(x):
    """
    构建卷积神经网络
    :param x: [-1, 20, 80, 3]
    :return:
    """
    # 1、第一个卷积大层
    # [-1, 20, 80, 3]----->[-1, 10, 40, 32]
    with tf.compat.v1.variable_scope("conv1"):
        # 卷积层
        conv1_weights = create_weights(shape=[5, 5, 3, 32])
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=x, filters=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(input=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2、第二个卷积大层
    # [-1, 10, 40, 32]----------->[-1, 5, 20, 64]
    with tf.compat.v1.variable_scope("conv2"):

        # 卷积层
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filters=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(input=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3、全连接层
    # [-1, 5, 20, 64]-------->[-1, 5*20*64]
    with tf.compat.v1.variable_scope("full_connection"):

        x_fc = tf.reshape(pool2_x, shape=[-1, 5*20*64])
        weights_fc = create_weights(shape=[5*20*64, 4*26])
        bias_fc = create_weights(shape=[4*26])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


if __name__ == '__main__':
    filename, image = read_picture()

    # 准备数据
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 20, 80, 3])
    y_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 4*26])

    # 构建模型
    y_predict = creat_model(x)

    # 构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)

    # 优化损失
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 计算准确率
    equal_list = tf.equal(tf.argmax(tf.reshape(y_true, shape=[-1, 4, 26]), axis=2),
                          tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 26]), axis=2))
    accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(equal_list, axis=1), dtype=tf.float32))

    # 初始化变量
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:

        # 运行初始化
        sess.run(init)

        # 开启线程
        # 创建线程协调员
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess, coord=coord)

        for i in range(1000):
            new_filename, new_image = sess.run([filename, image])
            labels = filename2lable(new_filename)

            # 将标签值转换为one-hot编码
            labels_value = tf.reshape(tf.one_hot(labels, depth=26), shape=[-1, 4*26]).eval()

            _, error, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={x: new_image, y_true: labels_value})
            print("第%d次训练后损失为%f，准确率为%f" % (i+1, error, accuracy_value))

        # 回收线程
        coord.request_stop()
        coord.join(threads)
