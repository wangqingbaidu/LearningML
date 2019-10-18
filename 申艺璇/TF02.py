# -*- coding: UTF-8 -*-
'''
Created on 2019��10��18��

@author: sheny
'''
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

sess = tf.Session()
inputs = tf.placeholder(tf.float32,[None,784])
fc1 = tf.contrib.layers.fully_connected(inputs,1024)
fc2 = tf.contrib.layers.fully_connected(fc1,2048)
fc3 = tf.contrib.layers.fully_connected(fc2,1024)
fc = tf.add(fc1,fc3)
fc4 = tf.contrib.layers.fully_connected(fc,512)
outputs = tf.contrib.layers.fully_connected(fc4,10,activation_fn=None)
softmax_outputs = tf.nn.softmax(outputs)
predictions = tf.argmax(softmax_outputs,axis=1)

merged = tf.summary.merge_all()
tf.summary.FileWriter('mnist_dnn',sess.graph)

sess.run(tf.global_variables_initializer())

random_inputs = np.random.rand(3,784)
print(sess.run(predictions,{inputs:random_inputs}))
