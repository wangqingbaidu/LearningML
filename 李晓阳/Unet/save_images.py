
# coding: utf-8

# In[ ]:


#用于UNet类中train函数中自动保存监测图片（所监测的batch_size中输出max_samples个图片对比查看）


# In[ ]:


import numpy as np
import cv2




#定义函数，通过imencode保存图片
def save_images(input_data,output1,output2,input_path,image_path,max_samples=4):
    """
    参数定义：
    """
    #横向拼接，将mask文件的输入和预测连接起来，[batch_size,324,648,1]
    image=np.concatenate([output1, output2], axis=2)
    #纵向拼接，max_samples个文件拼接到一起
    #当max_samples大于一个batch_size时，max_samples取到batch_size
    if max_samples > int(image.shape[0]):
        max_samples = int(image.shape[0])
    
    #取image中max_samples个图像
    image=image[0:max_samples,:,:,:]
    #四维化到三维，纵向拼接
    image=np.concatenate([image[i,:,:,:] for i in range(max_samples)],axis=0)
    image=np.uint8(image.clip(0., 1.) * 255.)
    
    #imread、imwrite不支持带有中文的路径
    #cv2.imwrite(image_path, image)
    """
    imwread可以用imdecode代替
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    也可以写成cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)；
    cv2.IMREAD_UNCHANGED参数可以用-1代替
    cv2.IMREAD_GRAYSCALE:以灰度模式读入图像：其值为0
    cv2.IMREAD_COLOR:读入彩色图像：其值为1；
    """
    cv2.imencode(".png",image)[1].tofile(image_path)
    """
    imwrite可以用imencode函数代替
    np.fromfile()函数相对应的函数为np.tofile()
    cv2.imencode()函数返回两个值;写入成功返回Ture，另一个值为数组.
    _,im_encode = cv2.imencode(".jpg",img)
    """
    
    #input不为None，还要保存input图片
    if input_data is not None:
        input_data=input_data[0:max_samples,:,:,:]
        input_data = np.concatenate([input_data[i, :, :, :] for i in range(max_samples)], axis=0)
        #cv2.imwrite(input_path, np.uint8(input_data.clip(0., 1.) * 255.))
        cv2.imencode(".png",np.uint8(input_data.clip(0., 1.) * 255.))[1].tofile(input_path)

