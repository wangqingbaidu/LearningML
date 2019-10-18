
# coding: utf-8



#功能函数的定义
import tensorflow as tf
import numpy as np


#卷积层
def conv(x,filter_height,filter_width,num_filters,stride_y,stride_x,name,padding='SAME',groups=1):
    #获得输入矩阵通道数
    input_channels=int(x.get_shape()[-1])
    #为卷积层创建函数，变量i、k分别代表输入矩阵和卷积核
    convolve=lambda i,k:tf.nn.conv2d(i,k,strides=[1,stride_y,stride_x,1],padding=padding)
    #定义命名空间
    with tf.variable_scope(name) as scope:
        #创建卷积层的权重和偏差
        weights=tf.get_variable('weights',shape=[filter_height,filter_width,input_channels/groups,num_filters])
        biases=tf.get_variable('biases',shape=[num_filters])
    #对groups进行设置，在原始AlexNet中，将输入分到了上下两层分别训练再融合，groups不为1，用多个CPU可以加速，但简单情况下我们设置为1
    if groups==1:
        conv=convolve(x,weights)
    #多个groups情况下，对输入和权重进行分解(axis=3，沿维度3分解)
    else:
        input_groups=tf.split(axis=3,num_or_size_splits=groups,value=x)
        weight_groups=tf.split(axis=3,num_or_size_splits=groups,value=weights)
        output_groups=[convolve(i,k) for i,k in zip(input_groups,weight_groups)]
        #将两个groups进行连接——在每次卷积后立即进行连接？？？
        conv=tf.concat(axis=3,values=output_groups)
    #添加偏差
    bias=tf.reshape(tf.nn.bias_add(conv,biases),tf.shape(conv))#进行tf.nn.bias_add操作后输出为array，所以要reshape一下
    #应用relu函数
    relu=tf.nn.relu(bias,name=scope.name)
    return relu


#全连接层
def fc(x,num_in,num_out,name,relu=True):
    with tf.variable_scope(name) as scope:
        #创建权重和偏差
        weights=tf.get_variable('weights',shape=[num_in,num_out],trainable=True)
        biases=tf.get_variable('biases',shape=[num_out],trainable=True)
        #输入矩阵乘以权重矩阵加上偏差
        act=tf.nn.xw_plus_b(x,weights,biases,name=scope.name)
    if relu:
        relu=tf.nn.relu(act)
        return relu
    else:
        return act



#池化层
def max_pool(x,filter_height,filter_width,stride_y,stride_x,name,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],
                          padding=padding,name=name)



#局部响应归一化
def lrn(x,radius,alpha,beta,name,bias=1.0):
    return tf.nn.local_response_normalization(x,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name=name)



#dropout
def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob)



#定义AlexNet网络
class AlexNet(object):
    #配置AlexNet网络
    def __init__(self,x,keep_prob,num_classes,skip_layer,weights_path='DEFAULT'):
        """x:输入
           keep_prob:dropout率
           num_classes:数据集标签类别个数
           skip_layer:List of names of the layer, that get trained from scratch
           weights_path:预训练的权重文件，当不与代码在同一文件夹下时需给出
        """
        #将输入参数解析为类变量
        self.X=x
        self.NUM_CLASSES=num_classes
        self.KEEP_PROB=keep_prob
        self.SKIP_LAYER=skip_layer
        
        if weights_path=='DEFAULT':
            self.WEIGHTS_PATH='alexnet.npy'
        else:
            self.WEIGHTS_PATH=weights_path
        #通过调用另一函数实现网络创建
        self.create()
    
    def create(self):
        #定义网络创建函数create
        
        #第一层：卷积——lrn归一化——池化
        conv1=conv(self.X,11,11,96,4,4,padding='VALID',name='conv1')
        norm1=lrn(conv1,2,1e-04,0.75,name='norm1')
        pool1=max_pool(norm1,3,3,2,2,padding='VALID',name='pool1')
        
        #第二层：卷积——lrn归一化——池化（原程序中采用了groups=2，这里仍采用默认值groups=1）
        conv2=conv(pool1,5,5,256,1,1,name='conv2')# padding=‘SAME’，output_size = input_szie / stride，向上取整
        norm2=lrn(conv2,2,1e-04,0.75,name='norm2')
        pool2=max_pool(norm2,3,3,2,2,padding='VALID',name='pool2')
        
        #第三层：卷积
        conv3=conv(pool2,3,3,384,1,1,name='conv3')
        
        #第四层：卷积
        conv4=conv(conv3,3,3,384,1,1,name='conv4')
        
        #第五层：卷积——池化
        conv5=conv(conv4,3,3,256,1,1,name='conv5')
        pool5=max_pool(conv5,3,3,2,2,padding='VALID',name='pool5')
        
        #第六层：拉平——全连接——dropout
        flattened=tf.reshape(pool5,[-1,6*6*256])#-1相当于未知数，会根据其他值自动变化，6*6为卷积后的大小
        fc6=fc(flattened,6*6*256,4096,name='fc6')
        dropout6=dropout(fc6,self.KEEP_PROB)  
        #用dropout drop掉部分输出（drop掉的为0值，其余变成1/keep_prob倍，用梯度反向传播时drop掉的值对应的权重就不会参与训练）
        
        #第七层：全连接——dropout
        fc7=fc(dropout6,4096,4096,name='fc7')
        dropout7=dropout(fc7,self.KEEP_PROB)
        
        #第八层：无激活函数全连接并返回到self
        self.fc8=fc(dropout7,4096,self.NUM_CLASSES,relu=False,name='fc8')
    
    def load_initial_weights(self,session):
        """载入权重到网络中
           由于 http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ 中的权重是以序列字典形式(e.g. weights['conv1'] is a list)
           而非字典字典形式(e.g. weights['conv1'] is a dict with keys 'weights' &'biases')，因此需要函数载入
        """
        #载入权重
        weights_dict=np.load(self.WEIGHTS_PATH,encoding='bytes').item()
        #循环载入所有的层
        for op_name in weights_dict:
            #检查该层是否需要从初始开始训练（需要从初始开始训练的层设到skip_layer里面）
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name,reuse=True):#允许命名空间再利用
                    #配置权重/偏差到对应的变量
                    for data in weights_dict[op_name]:
                        #偏差
                        if len(data.shape)==1:
                            var=tf.get_variable('biases',trainable=False)
                            session.run(var.assign(data))
                        else:
                            var=tf.get_variable('weights',trainable=False)
                            session.run(var.assign(data))


