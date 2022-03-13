#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/10 16:17
# @Author  : 小海腾

# 狗图片提取

import tensorflow as tf
import os


def picture_read(file_list):
    """
    狗图片读取
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    # 1、构造文件名队列
    file_queue = tf.compat.v1.train.string_input_producer(file_list)
    # 2、读取与解码
    reader = tf.compat.v1.WholeFileReader()
    # key是文件名，value是一个样本，也就是一张图片
    key, value = reader.read(file_queue)

    # 解码阶段
    image = tf.image.decode_jpeg(value)
    print("image", image)

    # 图像的形状类型修改
    image_resized = tf.compat.v1.image.resize_images(image, [200, 200])
    image_resized.set_shape([200, 200, 3])
    print("image_resized", image_resized)

    # 3、批处理
    image_batch = tf.compat.v1.train.batch([image_resized], 5, num_threads=1, capacity=10)
    print("image_batch", image_batch)

    with tf.compat.v1.Session() as sess:
        # 开启线程
        # 创建线程协调员
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess, coord=coord)

        new_key, new_value, new_image, new_image_resized, new_image_batch = sess.run([key, value, image, image_resized, image_batch])
        print("new_image_resized", new_image_resized)
        print("------------")
        print("new_image_batch", new_image_batch)

        # 回收线程
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # 构造路径+文件名列表
    filename = os.listdir("./dog")
    # 拼接路径+文件名
    file_list = [os.path.join("./dog/", file) for file in filename]
    picture_read(file_list)

