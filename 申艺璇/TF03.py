# -*- coding: UTF-8 -*-
'''
Created on 2019��10��18��

@author: sheny
'''
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random

batch_size = 32
GTy = lambda x:4*x*x+3*x+5
def gt(batch_size):
    x = []
    y = []
    for _ in range(batch_size):
        i = random.uniform(-1.0,1.0)
        x.append(i)
        y.append(GTy(i))
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

sess = tf.Session()
a = tf.get_variable('a',[],dtype=tf.float32,initializer=tf.random_uniform_initializer)
b = tf.get_variable('b',[],dtype=tf.float32,initializer=tf.random_uniform_initializer)
c = tf.get_variable('c',[],dtype=tf.float32,initializer=tf.random_uniform_initializer)

x = tf.placeholder(tf.float32,[batch_size])
labels = tf.placeholder(tf.float32,[batch_size])
y = a*x*x+b*x+c

loss = tf.losses.mean_squared_error(y,labels)
tf.summary.scalar('loss',loss)
#train_op = tf.train.RMSPropOptimizer(0.01).minimize(loss)
#train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('parabola',sess.graph)

sess.run(tf.initialize_all_variables())
for i in range(3000):
    x_batch,labels_batch = gt(batch_size)
    feed_back = sess.run([merged,train_op,loss,a,b,c],feed_dict={x:x_batch,labels:labels_batch})
    print('step:%.5d\tlosses:%6f\ta:%.6f\tb:%.6f\tc:%.6f'
          %(i,feed_back[2],feed_back[3],feed_back[4],feed_back[5]))
    writer.add_summary(feed_back[0],i)

    