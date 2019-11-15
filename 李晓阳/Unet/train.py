
# coding: utf-8


#训练和测试合一的运行代码



import tensorflow as tf
import numpy as np
import Unet_main
import cv2




#main函数
def main(argv):
    #tf.app.flags.FLAGS接受命令行传递参数或者tf.app.flags定义的默认参数
    tf_flags=tf.app.flags.FLAGS#tf_flags用在UNet类定义中，tf.app.flags.FLAGS 用来命令行运行代码时传递参数。
    
    #GPU配置——本TensorFlow还不能用GPU版
    #tf.ConfigProto()函数用在创建session时进行参数配置
    #config=tf.ConfigProto()
    
    #tf提供了两种控制GPU资源的使用方法
    #第一种是限制GPU使用率
    #config.gpu_options.per_process_gpu_memory_fraction=0.5 #占用50%显存
    #第二种是让TensorFlow在运行过程中动态申请显存，需要多少就申请多少
    #config.gpu_options.allow_growth=True
    
    config=tf.ConfigProto(allow_soft_placement=True)
    #with tf.device('/cpu:0'):
    #训练集
    if tf_flags.phase=="train":
        #使用上述定义的config配置session
        with tf.Session(config=config) as sess:
            #定义Unet类
            train_model=Unet_main.UNet(sess,tf_flags)
            #训练网络
            train_model.train(tf_flags.batch_size,tf_flags.training_steps,tf_flags.summary_steps,
                                tf_flags.checkpoint_steps,tf_flags.save_steps)
    #测试集,通过imwrite写入图片
    else:
        with tf.Session(config=config) as sess:
            #定义Unet类，这是读取吗，模型参数的基础
            test_model=Unet_main.UNet(sess,tf_flags)
            #测试，加载checkpoint文件参数
            test_model.load(tf_flags.checkpoint)
            image,output_masks=test_model.test()
            
            #保存图片，路径在该文件夹下
            filename_A="input.png"
            filename_B="output_masks.png"
            
            #uint8是无符号八位整型，表示范围是[0, 255]的整数
            """
            Clip（limit）the values in the array.这个方法会给出一个区间，在区间之外的数字将被剪除到区间的边缘，
            例如给定一个区间[0,1]，则小于0的将变成0，大于1则变成1.
            """
            #cv2.imwrite(filename_A,np.uint8(image[0].clip(0.,1.)*255))
            cv2.imencode(".png",np.uint8(image[0].clip(0.,1.)*255))[1].tofile(filename_A)
            #cv2.imwrite(filename_B,np.uint8(output_masks[0].clip(0.,1.)*255.))
            cv2.imencode(".png",np.uint8(output_masks[0].clip(0.,1.)*255.))[1].tofile(filename_B)
            #一定要转化为uint8格式吗？？？？？？？
            
            print("Saved files:{},{}".format(filename_A,filename_B))




#传递参数给tf.app.flags,可以用定义的方法给tf.app.flags一些默认参数，相当于接受python文件命令行执行时后面给的的参数
#tf.app.flags.DEFINE_string("job_name", "", "name of job") #参数名称、默认值、参数描述
"""
tf.app.flags.DEFINE_string():定义一个用于接收 string 类型数值的变量;
tf.app.flags.DEFINE_integer():定义一个用于接收 int 类型数值的变量;
tf.app.flags.DEFINE_float():定义一个用于接收 float 类型数值的变量;
tf.app.flags.DEFINE_boolean():定义一个用于接收 bool 类型数值的变量;
"""
#tf_flags=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')#不加这行会报错“UnrecognizedFlagError: Unknown command line flag 'f'”,也不知道为啥
tf.app.flags.DEFINE_string("output_dir","./dataset/out_test","checkpoint and summary directory")#保存模型和summary的输出文件夹
tf.app.flags.DEFINE_string("phase","train","model phase:train/test")#代码模式，训练/测试
tf.app.flags.DEFINE_string("training_set","./dataset/tfrecords","dataset path for training.")#存放tfrecords的文件夹
tf.app.flags.DEFINE_string("testing_set","./dataset/out_test/test","dataset path for testing one image pair.")#存放测试图片的文件夹
tf.app.flags.DEFINE_integer("batch_size",20,"batch size for training.")#UNet类中train函数输入batch_size
tf.app.flags.DEFINE_integer("training_steps",100,"total training steps.")#UNet类中train函数输入training_steps总训练步
tf.app.flags.DEFINE_integer("summary_steps",2,"summary period.")#UNet类中train函数输入summary_steps保存summary间隔步，可以设为1
tf.app.flags.DEFINE_integer("checkpoint_steps",50,"checkpoint period.")#UNet类中train函数输入checkpoint_steps保存模型间隔步
tf.app.flags.DEFINE_integer("save_steps",20,"checkpoint period.")#UNet类中train函数输入save_steps保存监测图片间隔步
tf.app.flags.DEFINE_string("checkpoint",None,"checkpoint name for restoring.")#load函数所载入的模型
#入口函数。通过处理flag解析，然后执行main函数
tf.app.run(main=main)

