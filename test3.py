#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/6 14:33
# @Author  : 小海腾

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 显示一部分图片数据
# plt.figure(figsize=(10, 10))
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

if os.path.exists('./model.h5'):
    model = tf.keras.models.load_model('./model.h5')
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    #
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    model.save('model.h5')

print('在测试集上评估')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('看看测试集测试结果')
predictions = model.predict(test_images)
print('预测值 = %i ; 正确值 = %i' % (np.argmax(predictions[0]), test_labels[0]))

print('从测试集取一个图片测试')
img = test_images[1]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
print(np.argmax(predictions_single[0]), test_labels[1])


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, [28, 28])
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


filepath = './3.png'
test_my_img = load_and_preprocess_image(filepath)
test_my_img = (np.expand_dims(test_my_img, 0))
my_result = model.predict(test_my_img)
print('自己的图片预测值 = %i ; 文件名 = ', (np.argmax(my_result[0]), filepath))

