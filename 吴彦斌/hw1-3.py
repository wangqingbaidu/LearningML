# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:02:37 2019

@author: WYB
"""
import os

def func(spec_directory):
    for i in os.listdir(spec_directory):                             #遍历目录下的数据
        print(os.path.join(spec_directory, i))                #输出路径信息
        if os.path.isdir(os.path.join(spec_directory,i)):     #如果数据是目录，就进行递归
            func(os.path.join(spec_directory,i))

func("F:\python\study")