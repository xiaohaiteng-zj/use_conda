#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/10 18:02
# @Author  : 小海腾

import tensorflow as tf
import os


class Cifar(object):

    def __init__(self):
        self.height = 32
        self.weight = 32
        self.channels = 3

        self.image_bytes = self.weight * self.height * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes

    def read_and_decode(self, file_list):
        tf.compat.v1.disable_eager_execution()

        # 1、构造文件名队列
        file_queue = tf.compat.v1.train.string_input_producer(file_list)

        # 2、读取与解码
        reader = tf.compat.v1.FixedLengthRecordReader(self.all_bytes)
        # key是文件名，value是一个样本，也就是一张图片
        key, value = reader.read(file_queue)
        print("key", key)
        print("value", value)
        # 解码阶段
        decoded = tf.compat.v1.decode_raw(value, tf.uint8)
        print("decoded", decoded)

        # 将目标值与特征值切片切开
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        print("label", label)
        print("image", image)

        # 调整图片形状
        image_reshape = tf.reshape(image, shape=[self.channels, self.height, self.weight])
        print("image_reshape", image_reshape)

        # 转置
        image_transpose = tf.transpose(image_reshape, [1, 2, 0])
        print("image_transpose", image_transpose)

        # 调整图像类型
        image_cast = tf.cast(image_transpose, tf.float32)

        # 3、批处理
        label_batch, image_batch = tf.compat.v1.train.batch([label, image_cast], 100, num_threads=1, capacity=100)
        print("label_batch", label_batch)
        print("image_batch", image_batch)

        with tf.compat.v1.Session() as sess:
            # 开启线程
            # 创建线程协调员
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess, coord=coord)

            new_key, new_value, new_decoded, new_label, new_image, new_image_reshape, new_image_transpose, new_label_batch, new_image_batch = sess.run(
                [key, value, decoded, label, image, image_reshape, image_transpose, label_batch, image_batch])
            # print("new_key", new_key)
            # print("new_value", new_value)
            # print("new_decoded", new_decoded)
            # print("new_label", new_label)
            # print("new_image", new_image)
            # print("new_image_reshape", new_image_reshape)
            # print("new_image_transpose", new_image_transpose)
            # print("new_label_batch", new_label_batch)
            # print("new_image_batch", new_image_batch)

            # 回收线程
            coord.request_stop()
            coord.join(threads)

        return new_image_batch, new_label_batch

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将样本的特征值和目标值一起写入tfrecords文件
        :param image:
        :param label:
        :return:
        """
        tf.compat.v1.disable_eager_execution()

        with tf.compat.v1.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
            # 循环构造example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                # print("image", image)
                # print("label", label)
                example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(feature={
                    "image": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[image])),
                    "label": tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[label]))
                }))
                # 将序列化后的example写入文件
                writer.write(example.SerializerToString())


if __name__ == '__main__':
    # 构造路径+文件名列表
    filename = os.listdir("./cifar-10-batches-py")
    # 拼接路径+文件名
    file_list = [os.path.join("./cifar-10-batches-py/", file) for file in filename if file[:4] == "data"]
    cifar = Cifar()
    image_batch, label_batch = cifar.read_and_decode(file_list)
    cifar.write_to_tfrecords(image_batch, label_batch)
