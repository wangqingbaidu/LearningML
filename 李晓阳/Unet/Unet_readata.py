
# coding: utf-8



#读取tfrecord数据
#有待根据培训的PPT进行修改




import tensorflow as tf
import os




class Read_TFRecords(object):
    def __init__(self,filename,batch_size=6,image_h=512,image_w=512,
                 image_c=1,num_threads=2,capacity_factor=2,min_after_dequeue=1):
        """
        filename:TFRecord文件路径
        image_h/image_w：图像高宽
        image_c：图像通道数
        num_threads：TFRecord文件加载线程
        capacity_factor：一个参数，与capacity有关；
                         capacity为指定队列的大小，队列中缓冲元素小于该值时从文件列表继续入队。从文件列表获取文件名很快，因此该值大小随意
        TFRecord数据的读取相当于先对数据名称列表进行处理、队列化、乱序等，再根据列表对数据进行处理和输出
        """
        self.filename=filename
        self.batch_size=batch_size
        self.image_h=image_h
        self.image_w=image_w
        self.image_c=image_c
        self.num_threads=num_threads
        self.capacity_factor = capacity_factor
        self.min_after_dequeue=min_after_dequeue
    #没有多线程读取数据——可能在训练过程中使用了多线程？？？？？？？
    def read(self):
        #从TFRecord读取数据
        #第一步：生成文件名队列——可以直接在这里进行乱序操作？？？？？？?
        filename_queue=tf.train.string_input_producer([self.filename])
        
        #第二步：调用tf.TFRecordReader创建读取器,Reader个数为1，可以研究一下多个Reader？？？？？？？
        reader=tf.TFRecordReader()
        #读取样例，返回serialized_example对象
        key,serialized_example=reader.read(filename_queue)
        
        #第三步：解析样例，调用tf.parse_single_example操作将Example缓冲区解析为张量字典
        features=tf.parse_single_example(serialized_example,
                                        features={
                                            "data":tf.FixedLenFeature([],tf.string),
                                            "label":tf.FixedLenFeature([],tf.string)
                                        })
        #第四步：对图像张量解码并进行resize、归一化处理等
        # tensorflow里面提供解码的函数有两个，tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，得到图像的像素值
        image_data=tf.image.decode_jpeg(features['data'],channels=self.image_c,name="decode_image")
        image_label=tf.image.decode_jpeg(features['label'],channels=self.image_c,name="decode_image")
        #图片resize到指定输入大小
        image_data = tf.image.resize_images(image_data, [self.image_h, self.image_w],method=tf.image.ResizeMethod.BICUBIC)
        #label大小调为与模型输出一致324*324
        image_label = tf.image.resize_images(image_label, [324, 324],method=tf.image.ResizeMethod.BICUBIC)
        #像素值类型转换为tf.float32，归一化
        image_data = tf.cast(image_data, tf.float32) / 255.0 # convert to float32
        image_label = tf.cast(image_label, tf.float32) / 255.0 # convert to float32

        #第五步：乱序操作，tf.train.shuffle_batch将训练集打乱，每次返回batch_size份数据
        input_data,input_masks=tf.train.shuffle_batch([image_data,image_label],batch_size=self.batch_size,
                                                     capacity=self.min_after_dequeue + self.capacity_factor * self.batch_size,
                                                     min_after_dequeue=self.min_after_dequeue,
                                                     num_threads=self.num_threads,
                                                     name='images')
        return input_data,input_masks

