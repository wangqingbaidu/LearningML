
最近尝试使用Unet代码处理自己的一些图像。通过Unet_dataset.ipynb将原始彩色图像处理成如图“输入.jpg”的灰度图，再连同“标签.jpg”一同写入到tfrecords文件中，通过Unet_readata.py、save_images.py、Unet_main.py以及train.ipynb对网络进行训练，但在训练过程中出现了以下几个问题：
1. batch_size一旦大于8，电脑出现卡死现象，怀疑是否和训练的cpu有关（目前安装的TensorFlow是CPU版的），或者使用GPU版TensorFlow会好一些？
2. 使用tf.nn.sigmoid_cross_entropy_with_logits或tf.squared_difference，loss函数均不下降，怀疑是否是batch_size 太小的原因，或者是训练集图片的原因？
3. 训练中batch_size=5,迭代400步监测采样得到的图片如图“train_0400.png”所示，左侧为训练结果，右侧为对应标签，两者显示的范围大小都不一致，怀疑是否是网络中某一步多裁减了一下，或者是因为还未收敛，训练结果显示的只是tf.concat所合并的上采样前的图像？
附：训练集图片类似于“输入.jpg”、“标签.jpg”，为夹杂小气泡的圆柱阵列，圆柱阵列的位置在不同图片中略有变动，共228张（是否训练集过小？？）
