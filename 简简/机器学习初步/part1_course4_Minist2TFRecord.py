# -*- coding: UTF-8 -*-
# @Time : 2019/10/20 19:41
# @Author : Janeasefor
# @Site :
# @File : test2.py
# @Software: PyCharm
import tensorflow as tf
import os
import traceback
from utils import process_image, ImageCoder

# 避免低版本下不必要警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 如果输入的数据不是list类型的，例如是一个标量，需要先转化成list类型。
def int64_feature (value):
    # Int类型的数据转化。
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature (value):
    # Byte类型数据转化，一般存放语音或者视频的原始二进制文件流。
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature (value):
    # Float类型数据转化。
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# 手写字符数据存放的位置，文件格式：label.index.jpg。
# file_dir = sys.argv[1]
file_dir = 'C:/Users/ysl-pc/Desktop/机器学习入门/part1_course4/hand_writing_storage'
# 输出TF record的位置，如果位置不存在，那么创建。
# output_dir = sys.argv[2]
output_dir = 'C:/Users/ysl-pc/Desktop/机器学习入门/part1_course4/TF_record'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 特征映射定义两个字段，`label`存放数据的标签，`data`存放的是数据。
feature = {'label': None, 'data': None}

# 定义Coder
coder = ImageCoder()

if os.path.exists(file_dir):
    # 创建一个TF record的writer。
    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, 'mnist_byte.tfreocrd'))
    for file_name in os.listdir(file_dir):
        # 过滤掉该目录下面非`.jpg`结尾的文件。
        if file_name.endswith('.jpg'):
            try:
                label, index, _ = file_name.split('.')
                # 把读入的图像转化成灰度图。
                image_encoded, _, _ = process_image(os.path.join(file_dir, file_name), coder)
                # 构造此样本的特征映射。
                feature['label'] = int64_feature(int(label))
                feature['data'] = bytes_feature(image_encoded)
                # 序列化之后写入对应的TF record。
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except:
                traceback.print_exc()
                print('Error while serializing %s.' % file_name)
else:
    print('File dir %s not exist!' % file_dir)
