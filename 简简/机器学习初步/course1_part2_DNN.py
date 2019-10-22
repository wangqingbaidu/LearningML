# -*- coding: utf-8 -*-
# @Time : 2019/10/15 19:57
# @Author : Janeasefor
# @Site : 
# @File : course_part2_DNN.py
# @Software: PyCharm
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()
inputs = tf.placeholder(tf.float32, [None, 768])
fc1 = tf.contrib.layers.fully_connected(inputs, 512)
fc2 = tf.contrib.layers.fully_connected(fc1, 1024)
fc3 = tf.contrib.layers.fully_connected(fc2, 512)
fc4 = tf.contrib.layers.fully_connected(fc1 + fc3, 256)
outputs = tf.contrib.layers.fully_connected(fc4, 10, activation_fn=None)

# 计算输⼊入数据在10个类别上的概率
softmax_outputs = tf.nn.softmax(outputs)

# 获取概率最⼤大的那个节点作为类别的预测结果。
predictions = tf.argmax(softmax_outputs, axis=1)

# 保存计算图。
merged = tf.summary.merge_all()
tf.summary.FileWriter('C:/Users/ysl-pc/Desktop/机器学习入门/course1', sess.graph)

# 随机初始化⽹网络参数。
sess.run(tf.global_variables_initializer())

# 随机⽣生成⼀一些数据，运⾏行行计算图，观察预测结果。
random_inputs = np.random.rand(3, 768)
print(sess.run(predictions, {inputs: random_inputs}))
