# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.placeholder(tf.float32,[])
b = tf.placeholder(tf.float32,[])
c = tf.placeholder(tf.float32,[])
x = tf.placeholder(tf.float32,[])

y = a*x**2+b*x+c

sess = tf.Session()

merged = tf.summary.merge_all()
tf.summary.FileWriter('TF01',sess.graph)
print(sess.run(y,{x:5,a:3,b:4,c:9}))