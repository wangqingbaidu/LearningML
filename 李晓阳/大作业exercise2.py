
# coding: utf-8

#给定一个CSV文件，将里面指定列的指定内容替换成另外的内容，并保存到指定文件夹
import pandas as pd
def col_repalce(input_file,spec_column,from_content,to_content,save_to):
    f=open(input_file)
    data=pd.read_csv(f)
    s=data[data[spec_column]==from_content].index.values
    for i in s:
        data.loc[i][spec_column]=to_content
    data.to_csv(save_to,index=False,encoding="utf_8_sig")

col_repalce("q49_p7.75_up.csv",'Nu',10,0,"g.csv")


    


#统计某个指定文件夹下的文件数量和文件夹数量，并且需要考虑隐藏文件。
import os
import win32file
import win32con
def file_sum(spec_directory,show_hidden=False):
    dirs_num=0
    dirs=[]
    files_num=0
    files=[]
    for i in os.listdir(spec_directory):
        flag=win32file.GetFileAttributes(spec_directory+'\\'+i)
        is_hiden = flag & win32con.FILE_ATTRIBUTE_HIDDEN
        if is_hiden and show_hidden==False:
            continue
        else:
            if os.path.isfile(spec_directory+i):
                files_num+=1
                files.append(i)
            else:
                dirs_num+=1
                dirs.append(i)
    print("文件总数为%d:"%files_num)
    print(files)
    print("文件夹总数为%d:"%dirs_num)
    print(dirs)

file_sum("E:\文件\工作文件\助教",show_hidden=True)   


#将作业2的内容封装成一个类，show方法可以把里面的内容打印出来。
class dir_con:
    def __init__(self, path):  
        self.path=path 
    def show(self):
        DIR=self.path
        dirs_num=0
        dirs=[]
        files_num=0
        files=[]
        for i in os.listdir(DIR):
            flag=win32file.GetFileAttributes(DIR+'\\'+i)
            is_hiden = flag & win32con.FILE_ATTRIBUTE_HIDDEN
            if is_hiden:
                continue
            else:
                if os.path.isfile(DIR+i):
                    files_num+=1
                    files.append(i)
                else:
                    dirs_num+=1
                    dirs.append(i)
        print("文件总数为%d:"%files_num)
        print(files)
        print("文件夹总数为%d:"%dirs_num)
        print(dirs)
x=dir_con("E:\文件\工作文件\助教")
x.show()

