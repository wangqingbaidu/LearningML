# -*- coding: gbk -*-
import os

path = "C:/Users/ysl-pc/Desktop/LearningML/简简"
files_num, dirs_num = 0, 0


def folder_file_collect (path):
    global files_num, dirs_num
    if not os.path.exists(path):
        return -1
    for root, dirs, names in os.walk(path):
        for filename in names:
            # print(os.path.join(root, filename))
            files_num += 1
        for dirname in dirs:
            # print(os.path.join(root, dirname))
            dirs_num += 1


if __name__ == '__main__':
    folder_file_collect(path)
print('文件总数为:', files_num, '文件夹总数为', dirs_num)
