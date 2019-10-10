# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:10:26 2019

@author: WYB
"""

import os
import win32file


def func(spec_directory, show_hidden=False):
    dirnum = 0  # 文件夹数量
    filenum = 0  # 文件数量（不计文件夹）
    num = 0  # 总文件数量（计文件夹）
    for root, dirs, files in os.walk(spec_directory):
        for name in dirs:
            dirnum = dirnum + 1
    if show_hidden == True:
        for fn in os.listdir(spec_directory):
            num = num + 1
    if show_hidden == False:
        for i in os.listdir(spec_directory):
            flag = win32file.GetFileAttributes(spec_directory + '\\' + i)
            if flag & 2 != 0:
                num = num
            else:
                num = num + 1
    filenum = num - dirnum
    print("文件夹个数:", dirnum)
    print("文件个数:", filenum)


func("F:\python\study", show_hidden=False)
