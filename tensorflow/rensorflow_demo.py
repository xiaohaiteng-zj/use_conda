#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/22 23:04
# @Author  : 小海腾
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def tensorflow_demo():
    """
    tensorflow的基本结构
    :return:
    """
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果：", c)

    tf.compat.v1.disable_eager_execution()

    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("tensorflow加法运算的结果是：", c_t)

    with tf.compat.v1.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value:", c_t_value)


def graph_demo():
    """
    图的演示
    :return:
    """
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果：", c)

    tf.compat.v1.disable_eager_execution()

    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("tensorflow加法运算的结果是：", c_t)

    # 查看默认图
    # 1、调用方法
    default_graph = tf.compat.v1.get_default_graph()
    print("默认图属性", default_graph)
    # 2、查看属性
    print("a_t的图属性：", a_t.graph)

    # 开启会话
    with tf.compat.v1.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value:", c_t_value)

    # 自定义图
    new_g = tf.Graph()
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new:", c_new)

    # 开启会话
    with tf.compat.v1.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run(c_new)
        print("c_new_value:", c_new_value)


def summary_demo():
    """
    可视化图
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = tf.add(a_t, b_t)
    print("tensorflow加法运算的结果是：", c_t)

    # 查看默认图
    # 1、调用方法
    default_graph = tf.compat.v1.get_default_graph()
    print("默认图属性", default_graph)
    # 2、查看属性
    print("a_t的图属性：", a_t.graph)

    # 开启会话
    with tf.compat.v1.Session() as sess:
        # c_t_value = sess.run(c_t)
        print("c_t_value::::::::::", c_t.eval())
        # 将图写入本地生成events文件
        tf.compat.v1.summary.FileWriter("./tmp/summary", graph=sess.graph)

    # 自定义图
    new_g = tf.Graph()
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new:", c_new)

    # 开启会话
    with tf.compat.v1.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run(c_new)
        print("c_new_value:", c_new_value)


def session_demo():
    """
    会话演示
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = tf.add(a_t, b_t)
    print("tensorflow加法运算的结果是：", c_t)

    # 开启会话
    with tf.compat.v1.Session() as sess:
        abc = sess.run([a_t, b_t, c_t])
        print("abc:", abc)


def placeholder_demo():
    """
    占位符
    :return:
    """
    tf.compat.v1.disable_eager_execution()
    a_ph = tf.compat.v1.placeholder(tf.float32)
    b_ph = tf.compat.v1.placeholder(tf.float32)
    c_ph = tf.add(a_ph, b_ph)
    print("a_ph:", a_ph)
    print("b_ph:", b_ph)
    print("c_ph:", c_ph)\

    with tf.compat.v1.Session() as sess:
        c_ph_value = sess.run(c_ph, feed_dict={a_ph: 3.6, b_ph: 2.1})
        print("c_ph_value:", c_ph_value)


def tensor_demo():
    """
    张量
    :return:
    """
    tensor1 = tf.constant(67.0)
    tensor2 = tf.constant([1, 2, 3])
    tensor3 = tf.constant([[1, 2, 3], [4, 5, 6]])
    print("tensor1", tensor1)
    print("tensor2", tensor2)
    print("tensor3", tensor3)

    # 张量类型的修改
    l_cast = tf.cast(tensor2, dtype=tf.float32)
    print("l_cast", l_cast)


def variable_demo():
    """
    变量
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.variable_scope("my_scope"):
        a = tf.Variable(initial_value=30)
        b = tf.Variable(initial_value=40)
        c = tf.add(a, b)
    print(a)
    print(b)
    print(c)

    # 初始化变量
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        # 运行初始化
        sess.run(init)

        a_value, b_value, c_value = sess.run([a, b, c])
        print("a_value", a_value)
        print("b_value", b_value)
        print("c_value", c_value)


def linear_regression():
    """
    tensorflow实现线性回归
    :return:
    """
    tf.compat.v1.disable_eager_execution()

    # 1.准备数据
    x = tf.compat.v1.random_normal(shape=[100, 1])
    y_true = tf.matmul(x, [[0.8]]) + 0.7
    # 2.构造模型
    # 2.1 用变量定义模型参数
    weights = tf.Variable(initial_value=tf.compat.v1.random_normal(shape=[1, 1]))
    bias = tf.Variable(initial_value=tf.compat.v1.random_normal(shape=[1, 1]))
    y_predict = tf.matmul(x, weights) + bias
    # 3.构造损失函数
    error = tf.reduce_mean(tf.square(y_predict - y_true))
    # 4.优化损失
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)


    # 收集变量
    tf.compat.v1.summary.scalar("error", error)
    tf.compat.v1.summary.histogram("weights", weights)
    tf.compat.v1.summary.histogram("bias", bias)

    # 合并变量
    merged = tf.compat.v1.summary.merge_all()

    # 创建saver对象，用于保存和加载模型
    saver = tf.compat.v1.train.Saver()

    ## 初始化变量
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        # 运行初始化
        sess.run(init)

        # 将图写入本地生成events文件
        file_writer = tf.compat.v1.summary.FileWriter("./tmp/linear", graph=sess.graph)

        print("训练前模型参数：权重%f, 偏置%f, 损失%f" % (weights.eval(), bias.eval(), error.eval()))
        # 开始训练
        for i in range(1000):
            sess.run(optimizer)

            # 运行合并变量的操作
            summary = sess.run(merged)
            # 将每次迭代后的变量写入事件文件
            file_writer.add_summary(summary, i)

            # 保存模型
            if i % 10 == 0:
                saver.save(sess, "./tmp/model/my_liner.ckpt")

        # # 加载模型
        # if os.path.exists("./tmp/model/checkpoint"):
        #     saver.restore(sess, "./tmp/model/my_liner.ckpt")

        print("训练后模型参数：权重%f, 偏置%f, 损失%f" % (weights.eval(), bias.eval(), error.eval()))


# 定义命令行参数
tf.compat.v1.disable_eager_execution()
tf.compat.v1.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
tf.compat.v1.app.flags.DEFINE_string("model_dir", "unknow", "模型保存的路径")

# 简化变量名
Flags = tf.compat.v1.app.Flags


def command_demo():
    """
    命令行参数演示
    :return:
    """
    print("max_step", Flags.max_step)
    print("model_dir", Flags.model_dir)


if __name__ == '__main__':
    # tensorflow_demo()
    # graph_demo()
    # summary_demo()
    # session_demo()
    # placeholder_demo()
    # tensor_demo()
    # variable_demo()
    # linear_regression()
    command_demo()
