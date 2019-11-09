# -*- coding: UTF-8 -*-
'''
Created on 2019��10��22��

@author: sheny
'''
import tensorflow as tf
import os 
from tensorboard.plugins.debugger.constants import ALERT_REGISTRY_BACKUP_FILE_NAME
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import sys, traceback
from PIL import Image
import random

def int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

file_dir = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
feature = {'label':None,'data':None}

if os.path.exists(file_dir):
    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir,'mnist.tfrecord'))
    train_examples = [l for l in os.listdir(file_dir)]
    random.shuffle(train_examples)
    for file_name in train_examples:
        if file_name.endswith('.jpg'):
            try:
                label,index,_ = file_name.split('.')
                image_data = tf.gfile.FastGFile(os.path.join(file_dir,file_name),'rb').read()
             #   image_data = Image.open(os.path.join(file_dir,file_name)).convert('L')
             #   image_data_norm = (np.array(image_data,dtype=float)-127)/128.0
             #   image_data_norm_vector = np.reshape(image_data_norm,[-1])
                feature['label'] = int64_feature(int(label))
                feature['data'] = bytes_feature(image_data)
             #   feature['data'] = bytes_feature(image_data_norm_vector.tolist())
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except:
                traceback.print_exc()
                print('Error while serializing %s',file_name)
else:
    print('File dir %s not exist!',file_dir)

