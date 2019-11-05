
# coding: utf-8




#Unet网络定义和类定义





import os
import logging
import time
from datetime import datetime
import tensorflow as tf
from save_images import save_images
import sys
#sys.path.append("../data")
import Unet_readata
import numpy as np
import cv2




#定义Unet网络函数——所有层共用一个scope name,也就是不详细记录各层的信息？？？？？？？
def Unet(name,in_data,reuse=False):
    #确认输入数据非空。assert condition：用来让程序测试这个condition，如果condition为false，那么raise一个AssertionError出来。
    assert in_data is not None
    #用tf.variable_scope管理范围
    with tf.variable_scope(name,reuse=reuse):
        #每经过一个卷积层对特征图裁剪备用，以便上采样之后的合并
        
        #第一层
        conv1_1=tf.layers.conv2d(in_data,64,3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        ## kernel_initializer = tf.variance_scaling_initializer(scale=2.0))
        """
        tf.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
            )
            相比于tf.nn.conv2d,tf.layers.conv2d为高级封装函数，通过一个函数实现了卷积、偏差、激活、初始化等全部过程
        """
        #第二次卷积
        conv1_2=tf.layers.conv2d(conv1_1,64,3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        #第一次修建备用,上下左右各裁剪90，[None,508,508,64]——[None,328,328,64]
        crop1=tf.keras.layers.Cropping2D(cropping=((90,90),(90,90)))(conv1_2)
        #池化层,size=[None,508,508,64]==>[None,254,254,64]
        pool1=tf.layers.max_pooling2d(conv1_2,2,2)
        """
        tf.layers.max_pooling2d(
            inputs,
            pool_size,
            strides,
            padding='valid',
            data_format='channels_last',一个字符串,表示输入中维度的顺序.支持channels_last(默认)和channels_first；
                channels_last对应于具有形状(batch, height, width, channels)的输入,
                而channels_first对应于具有形状(batch, channels, height, width)的输入.
            name=None
            )
        """
        
        #第二层
        conv2_1=tf.layers.conv2d(pool1,128,3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_2=tf.layers.conv2d(conv2_1,128,3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        #裁剪,[None,250,250,128]——[None,168,168,128]
        crop2=tf.keras.layers.Cropping2D(cropping=((41,41),(41,41)))(conv2_2)
        #池化
        pool2=tf.layers.max_pooling2d(conv2_2,2,2)
        
        #第三层
        conv3_1=tf.layers.conv2d(pool2,256,3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3_2=tf.layers.conv2d(conv3_1,256,3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        #裁剪size=[None,121,121,256]==>[None,88,88,256]
        crop3=tf.keras.layers.Cropping2D(cropping=((16,17),(16,17)))(conv3_2)
        #池化
        pool3=tf.layers.max_pooling2d(conv3_2,2,2)
        
        #第四层
        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #dropout
        drop4=tf.layers.dropout(conv4_2)
        """
        tf.layers.dropout(
            inputs,
            rate=0.5,
            noise_shape=None,
            seed=None,
            training=False,
            name=None
            )
        """
        #裁剪[None,56,56,512]==>[None,48,48,512]
        crop4=tf.keras.layers.Cropping2D(cropping=((4,4),(4,4)))(drop4)
        #池化[None,56,56,512]==>[None,28,28,512]
        pool4 = tf.layers.max_pooling2d(drop4, 2, 2)
        
        #第五层
        conv5_1 = tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #dropout
        drop5=tf.layers.dropout(conv5_2)
        
        #解码器开始
        #第六层：
        #上采样扩充：[None,24,24,1024]==>[None,48,48,1024]
        up6_1=tf.keras.layers.UpSampling2D(size=(2,2))(drop5)
        #tf.nn.conv2d_transpose
        """
        keras.layers.UpSampling2D(
            size=(2, 2), 沿着数据的行和列分别重复 size[0] 和 size[1] 次，对于本例即对矩阵中的每个数据横向、纵向复制为两个
            data_format=None, 
            interpolation='nearest')
        """
        #卷积降维：1024——512
        up6 = tf.layers.conv2d(up6_1, 512, 2, padding="SAME", activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #和编码过程中剪裁好备用的feature map合并
        merge6=tf.concat([crop4,up6],axis=3)
        #卷积操作
        conv6_1 = tf.layers.conv2d(merge6, 512, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        
        #第七层
        #上采样：[None,44,44,512]==>[None,88,88,512]
        up7_1 = tf.keras.layers.UpSampling2D(size=(2,2))(conv6_2)
        #卷积降维到[None,88,88,256]
        up7 = tf.layers.conv2d(up7_1, 256, 2, padding="SAME", activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #合并
        merge7 = tf.concat([crop3, up7], axis=3)
        #卷积
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        
        #第八层
        #上采样：[None,84,84,256]==[None,168,168,256]
        up8_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        #卷积降维——128
        up8 = tf.layers.conv2d(up8_1, 128, 2, padding="SAME", activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #合并
        merge8 = tf.concat([crop2, up8], axis=3)
        #卷积
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        
        #第九层
        #上采样：[None,164,164,128]==>[None,328,328,128]
        up9_1 = tf.keras.layers.UpSampling2D(size=(2,2))(conv8_2)
        #卷积降维——64
        up9 = tf.layers.conv2d(up9_1, 64, 2, padding="SAME", activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #合并
        merge9 = tf.concat([crop1, up9], axis=3)
        #卷积
        conv9_1 = tf.layers.conv2d(merge9, 64, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv9_2 = tf.layers.conv2d(conv9_1, 64, 3, activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        #卷积降维到2层
        conv9_3 = tf.layers.conv2d(conv9_2, 2, 3, padding="SAME", activation=tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer())
        
        #第十层：[None,324,324,2]=>[None,324,324,1]
        conv10 = tf.layers.conv2d(conv9_3, 1, 1,kernel_initializer = tf.contrib.layers.xavier_initializer())
        
    return conv10




#结合输入定义UNet模型类,这里采用大写N以区别Unet函数
class UNet(object):
    def __init__(self,sess,tf_flags):#tf_flags,在train函数中定义
        self.sess=sess#tf.Session()
        self.dtype=tf.float32
        
        #模型保存位置
        self.output_dir=tf_flags.output_dir
        #checkpoint保存位置
        self.checkpoint_dir=os.path.join(self.output_dir,"checkpoint")
        #checkpoint保存文件名
        self.checkpoint_prefix="model"
        self.saver_name="checkpoint"
        #summary文件保存目录
        self.summary_dir=os.path.join(self.output_dir,"summary")
        
        self.is_training=(tf_flags.phase=="train")
        
        #初始化学习率,此学习率作为初参数
        self.learning_rate=0.001
        
        #数据参数，图像size 512*512*1
        self.image_w=512
        self.image_h=512
        self.image_c=1
        
        #输入placeholder
        self.input_data=tf.placeholder(self.dtype,[None,self.image_h,self.image_w,self.image_c])
        #mask placeholder
        self.input_mask=tf.placeholder(self.dtype,[None,324,324,self.image_c])
        
        #学习率占位符，此占位符用于最后输入优化器中的实际学习率参数
        self.lr=tf.placeholder(self.dtype)#此学习率用于定义优化器了
        
        #针对train
        if self.is_training:
            #训练集目录
            self.training_set=tf_flags.training_set
            #训练过程中保存样本进行监测的目录
            self.sample_dir=os.path.join(self.output_dir,"train_results")
            
            #创建summary_dir,checkpoint_dir,sample_dir
            self._make_aux_dirs()#_make_aux_dirs函数见下文
            
            #定义loss，优化器，summary，saver
            self._build_training()#_build_training函数见下文
            
            #设定运行过程中的输出日志格式、名称、路径等
            log_file=self.output_dir+"/Unet.log"
            logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                                filename=log_file,level=logging.DEBUG,filemode='w')
            """
            level一共分成5个等级，从低到高分别是：DEBUG ,INFO ,WARNING ,ERROR, CRITICAL,DEBUG输出详细的信息,通常只出现在诊断问题上
            filename: 指定日志文件名
            filemode: 和file函数意义相同，指定日志文件的打开模式，'w'或'a'
            format: 指定输出的格式和内容，format可以输出很多有用信息，如上例所示:
                %(levelno)s: 打印日志级别的数值
                %(levelname)s: 打印日志级别名称
                %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
                %(filename)s: 打印当前执行程序名
                %(funcName)s: 打印日志的当前函数
                %(lineno)d: 打印日志的当前行号
                %(asctime)s: 打印日志的时间
                %(thread)d: 打印线程ID
                %(threadName)s: 打印线程名称
                %(process)d: 打印进程ID
                %(message)s: 打印日志信息
            datefmt: 指定时间格式，同time.strftime()
            level: 一共分成5个等级，从低到高分别是：DEBUG ,INFO ,WARNING ,ERROR, CRITICAL,
                设为DEBUG则级别高于DEBUG的内容会输出，默认为logging.WARNING
            stream: 指定将日志的输出流，可以指定输出到sys.stderr,sys.stdout或者文件，
                默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略
            """
            #logging.getLogger()创建一个记录器，addHandler()添加一个StreamHandler处理器
            logging.getLogger().addHandler(logging.StreamHandler())
        
        #对于测试集
        else:
            #测试集目录
            self.testing_set=tf_flags.testing_set
            #测试集输出
            self.output=self._build_set()#_build_set函数见下文
        
    #定义训练所需的loss，优化器，summary，saver
    def _build_training(self):
        #定义self.loss,self.opt,self.summary,self.writer,self.saver
            
        #通过调用Unet函数定义输出结果
        self.output=Unet(name="Unet",in_data=self.input_data,reuse=False)
            
        #定义loss函数,tf.reduce_mean求平均,用sigmoid交叉熵函数？？？？？？？？？？
        #self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_mask,logits=self.output))
        self.loss = tf.reduce_mean(tf.squared_difference(self.input_mask,self.output))
        #self.loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.input_mask, self.output))，适用于二分类
            
        #定义优化器
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,name="opt")#优化器
        """
        自适应矩估计（Adaptive Moment Estimation），也是梯度下降算法的一种变形，但是每次迭代参数的学习率都有一定的范围，
        不会因为梯度很大而导致学习率（步长）也变得很大，参数的值相对比较稳定。
        概率论中矩的含义是：如果一个随机变量 X 服从某个分布，X 的一阶矩是 E(X)，也就是样本平均值，X 的二阶矩就是 E(X^2)，也就是样本平方的平均值。
        Adam 算法利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。TensorFlow提供的tf.train.AdamOptimizer可控制学习速度，
        经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳
        """
        #定义写入summary
        tf.summary.scalar("loss",self.loss)
            
        self.summary=tf.summary.merge_all()
        self.writer=tf.summary.FileWriter(self.summary_dir,graph=self.sess.graph)
            
        #定义保存checkpoint
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()#用在写入日志时的_print_summary函数中（见下文），帮助解析loss summary的值
        
    def train(self,batch_size,training_steps,summary_steps,checkpoint_steps,save_steps):
        """
        参数：
        batch_size:
        training_steps:训练要经过多少迭代步
        summary_steps:每经过多少步就保存一次summary
        checkpoint_steps:每经过多少步就保存一次checkpoint文件
        save_steps:每经过多少步就保存一次图像
        """
        step_num=0
            
        #读取最近保存的参数变量
        latest_checkpoint=tf.train.latest_checkpoint(self.checkpoint_dir)
        
        #存在已经保存的参数变量且读取成功
        if latest_checkpoint:
            #记录模型已训练的时间步
            step_num=int(os.path.basename(latest_checkpoint).split("-")[1])#checkpoint文件的命名格式为model-10000.index
            assert step_num>0,"Please ensure checkpoint format is model-*.*." #若时间步没有记录成功，查看命名格式是否有误
                
            #加载使用最新保存的checkpoint文件
            self.saver.restore(self.sess,latest_checkpoint)
            #写入日志文件
            logging.info("{}:Resume training from step {}.Loaded checkpoint {}".format(datetime.now(),step_num,latest_checkpoint))
        #若不存在保存好的训练文件，则初始化
        else:
            self.sess.run(tf.global_variables_initializer())#初始化所有参数——那Unet网络中采用的初始化方法还有效吗？？？？？？？
            #写入日志文件
            logging.info("{}:Init new training".format(datetime.now()))
            
        #调用Read_TFRecords类对象读取数据 [batch_size,512,512,1],[batch_size,324,324,1]
        tf_reader=Unet_readata.Read_TFRecords(filename=os.path.join(self.training_set,"image.tfrecords"),
                                                 batch_size=batch_size,image_h=self.image_h,image_w=self.image_w,image_c=self.image_c)
        images,images_mask=tf_reader.read()
        

        #写入日志
        logging.info("{}:Done init data generators".format(datetime.now()))
            
        #启用线程协调器
        self.coord=tf.train.Coordinator()
        #使用tf.train.start_queue_runners之后，才会真正启动填充队列的线程。此后计算单元就可以拿到数据并进行计算
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        #在会话中运行训练
        try:
            """
            如果当try后的语句执行时发生异常，python就跳回到try并执行第一个匹配该异常的except子句，
            异常处理完毕，控制流就通过整个try语句（除非在处理异常时又引发新的异常）。
            如果在try子句执行时没有发生异常，python将执行else语句后的语句（如果有else的话），然后控制流通过整个try语句。
            无论try语句中是否抛出异常，finally中的语句一定会被执行
            """
            
            #开始训练
            c_time=time.time()#开始训练时间
            lrval=self.learning_rate#对学习率的解释见前述注释
            #注意如果之前有读取已训练的模型。step_num已经初始化为已训练完的迭代步
            for c_step in range(step_num+1,training_steps+1):
                #5000个step后，学习率减半
                if c_step%5000==0:
                    lrval=self.learning_rate*0.5
                    
                #循环读取TFRecord中的数据
                #b_time=time.time()
                batch_images,batch_images_masks=self.sess.run([images,images_mask])#不需要用Iterator迭代器？？？？？？？
                
                #实现反向传播优化
                #dq_time=time.time()-b_time#读取数据耗时
                #logging.info("{}:{}".format('读取数据',dq_time))
                #b_time=time.time()
                c_feed_dict={
                    self.input_data:batch_images,
                    self.input_mask:batch_images_masks,
                    self.lr:lrval
                }
                self.sess.run(self.opt,feed_dict=c_feed_dict)
                #yh_time=time.time()-b_time#优化模型耗时
                #logging.info("{}:{}".format('运行模型',yh_time))
                #保存summary
                if c_step%summary_steps==0:
                    #b_time=time.time()
                    c_summary=self.sess.run(self.summary,feed_dict=c_feed_dict)
                    self.writer.add_summary(c_summary,c_step)
                        
                    e_time=time.time()-c_time#该summary的总时长
                    time_periter=e_time/summary_steps#平均每步迭代时长
                    #写入日志
                    logging.info("{}:Iteration_{}({:.4f}s/iter){}".format(datetime.now(),c_step,time_periter,
                                                                              self._print_summary(c_summary)))#_print_summary函数见下文
                    #su_time=time.time()-b_time
                    #logging.info("{}:{}".format('保存summary',su_time))
                    c_time=time.time()#重新计算起始时间
                    
                #保存模型 checkpoint
                if c_step%checkpoint_steps==0:
                    #b_time=time.time()
                    self.saver.save(self.sess,os.path.join(self.checkpoint_dir,self.checkpoint_prefix),global_step=c_step)
                    #写入日志
                    logging.info("{}:Iteration_{}Saved checkpoint".format(datetime.now(),c_step))
                    #ch_time=time.time()-b_time
                    #logging.info("{}:{}".format('保存checkpoint',ch_time))
                #保存图片
                if c_step%save_steps==0:
                    #b_time=time.time()
                    #得到预测的分割mask和ground truth的mask
                    _, output_masks, input_masks = self.sess.run([self.input_data, self.output, self.input_mask],
                                                                     feed_dict=c_feed_dict)#此处可以把input_data去掉吗？？？？？？？
                    #save_images函数见save_images.py文件，另保存位置可以更改一下
                    save_images(None, output_masks, input_masks,
                                #self.sample_dir：train_results
                                input_path = '{}/input_{:04d}.png'.format(self.sample_dir, c_step),
                                image_path = '{}/train_{:04d}.png'.format(self.sample_dir, c_step))
                    #sa_time=time.time()-b_time
                    #logging.info("{}:{}".format('保存sample',sa_time))
        except KeyboardInterrupt:#当错误类型为KeyboardInterrupt，即用户中断执行时
            print('Interrupted')
            self.coord.request_stop()# 通知线程停止
        except Exception as e:#错误类型为常规错误的基类
            self.coord.request_stop(e)# 将异常抛给coordinator，通知线程停止
        finally:
            self.coord.request_stop()#主线程计算完成，停止所有采集数据的进程
            self.coord.join(threads)#等待其他线程结束
        #写入日志
        logging.info("{}: Done training".format(datetime.now()))
    
    #定义测试集的输出结果
    def _build_test(self):
        output = Unet(name="UNet",in_data=self.input_data,reuse=False)
        #定义saver，用于读取保存的模型
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        #返回输出结果
        return output
    
    #加载训练模型，主要用于测试集
    def load(self,checkpoint_name=None):
        print("{}:Loading checkpoint...".format(datetime.now()))
        #读取指定保存节点的模型
        if checkpoint_name:
            checkpoint=os.path.join(self.checkpoint_dir,checkpoint_name)
            self.saver.restore(self.sess,checkpoint)
            print("loaded{}".format(checkpoint_name))
        #不指定节点，读取最近保存的模型
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:          
                #加载使用最新保存的checkpoint文件
                self.saver.restore(self.sess,latest_checkpoint)
                print("loaded{}".format(os.path.basename(latest_checkpoint)))
            else:
                #未读取到文件，使用raise抛出异常
                raise IOError("No checkpoints found in {}".format(self.checkpoint_dir))
    
    #定义测试函数
    def test(self):
        #只对一张图片测试
        image_name=glob.glob(os.path.join(self.testing_set,"*.jpg"))
        
        #读取图像数据
        # tensorflow中测试图片必须为255.0
        # OpenCV读取图片.图片格式为BGR, w.t，默认加载方式为channel=3，此处channel=0？？？？？？？
        #imread图片、resize图片到输入大小，reshape图片到模型所需的四个维度并除以255归一化——在这一步处理成了灰度图像？？？？？？？
        image = np.reshape(cv2.resize(cv2.imread(image_name[0], 0), (self.image_h, self.image_w)), (1, self.image_h, self.image_w, self.image_c)) / 255.
        print("{}: Done init data generators".format(datetime.now()))
        
        #输入数据得到结果
        # image: 1 * 512 * 512 * 1
        # output_masks: 1 * 324 * 342 * 1.
        c_feed_dict = {self.input_data: image}
        output_masks = self.sess.run(self.output, feed_dict=c_feed_dict)
        return image, output_masks
    
    #创建summary_dir,checkpoint_dir,sample_dir
    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)   
    
    #写入日志时用到的_print_summary函数
    def _print_summary(self,summary_string):
        #解析loss summary中的值
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result) #使用.join函数，用空格连接result列表中的项




"""
#tf.keras.layers.UpSampling2D的一个验证
from tensorflow.keras.layers import UpSampling2D
import numpy as np
import tensorflow as tf
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
x=x.reshape(1,4,4,1)
x=tf.convert_to_tensor(x)
y=UpSampling2D(size=(1,1))(x)
with tf.Session() as sess:
    print(y.eval())
"""

