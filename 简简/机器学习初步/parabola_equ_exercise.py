# -*- coding: utf-8 -*-
# @Time : 2019/10/16 19:22
# @Author : Janessefor
# @Site : 
# @File : parabola_equ_exercise.py
# @Software: PyCharm
# 设置范围为-4..4,20组练习5000次，优化方法选择Adam
import tensorflow as tf
import numpy as np

batch_size = 20
# 要拟合的函数表达式。
GTy = lambda x: 1 * x * x + 2 * x + 1


# ⽣生成⼀一个批量量的训练数据。
def gt(batch_size):
    x = np.array((np.random.rand(batch_size)))
    x = (x * 2 - 1) * 4
    y = list(map(GTy, x))
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


sess = tf.Session()
w_a = tf.get_variable("w_a", [], dtype=tf.float32, initializer=tf.random_uniform_initializer)
w_b = tf.get_variable("w_b", [], dtype=tf.float32, initializer=tf.random_uniform_initializer)
w_c = tf.get_variable("w_c", [], dtype=tf.float32, initializer=tf.random_uniform_initializer)

# 创建训练数据的占位符。
x = tf.placeholder(tf.float32, [batch_size])
labels = tf.placeholder(tf.float32, [batch_size])
# 创建⼆二次抛物线函数的计算图。
y = w_a * x * x + w_b * x + w_c
loss = tf.losses.mean_squared_error(y, labels)
# 将loss添加到tensorboard。
tf.summary.scalar('loss', loss)
# 创建优化函数
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
# 汇总所有待写的数据
merged = tf.summary.merge_all()
# 创建一个summary writer
writer = tf.summary.FileWriter('C:/Users/ysl-pc/Desktop/机器学习入门/Part1_course/ML/part3', sess.graph)

# 初始化模型参数
sess.run(tf.initialize_all_variables())
for i in range(5000):
    x_batch, labels_batch = gt(batch_size)
    # feed_back取回对应的summary、loss、w_a、w_b、w_c。
    feed_back = sess.run([merged, train_op, loss, w_a, w_b, w_c],
                         feed_dict={x: x_batch, labels: labels_batch})
    print('step:%.5d\tlosses: %.6f\tw_a:%.6f\tw_b:%.6f\tW_c:%.6f'
          % (i, feed_back[2], feed_back[3], feed_back[4], feed_back[5]))
    # 写入到对应log文件。
    writer.add_summary(feed_back[0], i)
# tensorboard --logdir folder_name
