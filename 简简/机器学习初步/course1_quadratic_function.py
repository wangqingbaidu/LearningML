# -*- coding: UTF-8 -*-

import tensorflow as tf

# 初始化
sess = tf.Session()
a = tf.placeholder(tf.float32, [])
b = tf.placeholder(tf.float32, [])
c = tf.placeholder(tf.float32, [])
x = tf.placeholder(tf.float32, [])
y = a * x ** 2 + b * x + c
merged = tf.summary.merge_all()
tf.summary.FileWriter('C:/Users/ysl-pc/Desktop/机器学习入门/course1', sess.graph)
#  运行计算图
print(sess.run(y, feed_dict={a: 1, b: 2, c: 1, x: -1}))
