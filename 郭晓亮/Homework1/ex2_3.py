# -*- coding: utf-8 -*-
"""
Created on 2019-10-10 23:41:17
@author: GUO Xiaoliang
"""

import os
import win32file


class ShowFilesDirs():
    def __init__(self, spec_directory):#, show_hidden):
        self.spec_directory = spec_directory

    def get_dirs_lists(self):
        self.list_all = os.listdir()
        if self.show_hidden == True:
            self.list_dirs = []
            self.list_files = []
            self.list_dirs_hidden = []
            self.list_files_hidden = []

            for item in self.list_all:
                flag = win32file.GetFileAttributesW(item)
                if flag&2!=0:
                    if os.path.isdir(item):
                        self.list_dirs_hidden.append(item)
                    elif os.path.isfile(item):
                        self.list_files_hidden.append(item)
                else:
                    if os.path.isdir(item):
                        self.list_dirs.append(item)
                    elif os.path.isfile(item):
                        self.list_files.append(item)
        else:
            self.list_dirs = [item for item in self.list_all if os.path.isdir(item)]
            self.list_files = [item for item in self.list_all if os.path.isfile(item)]

    def show(self, show_hidden=False):
        self.show_hidden = show_hidden
        self.get_dirs_lists()
        if self.show_hidden == True:
            print('Non-hidden dirs: ', self.list_dirs) 
            print('Non-hidden files: ',self.list_files) 
            print('Hidden dirs: ',self.list_dirs_hidden)
            print('Hidden files: ',self.list_files_hidden)
        else:
            print('All dirs: ', self.list_dirs) 
            print('All files: ',self.list_files) 


    


if __name__=='__main__':
    cwd = os.getcwd()
    s = ShowFilesDirs(cwd)
    s.show(show_hidden=True)