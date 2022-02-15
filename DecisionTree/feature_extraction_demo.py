#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/10 17:02
# @Author  : 小海腾

# 特征提取

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def dict_demo():
    """
    字典特征提取
    :return:
    """
    # 1、获取数据
    data = [{"city": "北京", "temperature": 100},
            {"city": "上海", "temperature": 60},
            {"city": "深圳", "temperature": 30}]

    # sparse矩阵的优势：1、节省空间；2、提高读取效率
    transfer = DictVectorizer(sparse=True)
    data = transfer.fit_transform(data)
    print(data)

    names = transfer.get_feature_names()
    print("属性名字是：\n", names)


def english_text_demo():
    """
    英文文本特征提取
    查找文本中每个字符串出现的次数
    :return:
    """
    data = ["life is short, i like python",
            "life is too long, i dislike python"]

    # stop_words参数，说明哪个字符串不参与查找
    transfer = CountVectorizer(stop_words=["dislike"])
    new_data = transfer.fit_transform(data)
    print(new_data)
    print(new_data.toarray())

    names = transfer.get_feature_names()
    print("属性名字是：\n", names)


def chinese_text_demo():
    """
    中文文本特征提取
    查找文本中每个字符串出现的次数
    :return:
    """
    data = ["人生 苦短，我 喜欢 python",
            "生活 太长久，我 不喜欢 python"]

    # stop_words参数，说明哪个字符串不参与查找
    transfer = CountVectorizer()
    new_data = transfer.fit_transform(data)
    print(new_data)
    print(new_data.toarray())

    names = transfer.get_feature_names()
    print("属性名字是：\n", names)


def cut_world(text):
    """
    利用jieba对中文进行分词
    :return:
    """
    text = " ".join(list(jieba.cut(text)))

    return text


def tfidf_demo():
    """
    中文文本特征提取
    查找文本中每个字符串出现的次数
    :return:
    """
    data = ["人生 苦短，我 喜欢 喜欢 python",
            "生活 太长久，我 不喜欢 python"]

    # stop_words参数，说明哪个字符串不参与查找
    # transfer = CountVectorizer()
    transfer = TfidfVectorizer()
    new_data = transfer.fit_transform(data)
    print(new_data)
    print(new_data.toarray())

    names = transfer.get_feature_names()
    print("属性名字是：\n", names)


if __name__ == '__main__':
    # dict_demo()
    # english_text_demo()
    # chinese_text_demo()
    # re = cut_world("我爱天安门")
    # print(re)
    tfidf_demo()