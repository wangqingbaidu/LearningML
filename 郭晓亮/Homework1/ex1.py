# -*- coding: utf-8 -*-
"""
Created on 2019-10-10 21:12:54
@author: GUO Xiaoliang
"""

import os 
import pandas as pd
import numpy as np
import time

def change_col_save_to_new_file(input_file, spec_column, from_content, to_content, save_to):
    data = pd.read_csv(input_file)
    print('First three rows:')
    print(data.head(3))

    start = time.time()
    # # 法1：for循环寻找对应值
    # for i in range(data.shape[0]):
    #     if data.iloc[i][spec_column] == from_content:
    #         data.iloc[i][spec_column] = to_content

    # 法2：使用isin和at函数直接确定该值位置
    if type(from_content) is list:
        flag = data[spec_column].isin(from_content)
    elif type(from_content) is float:
        flag = data[spec_column].isin([from_content])
    data.at[flag, spec_column] = to_content
    end = time.time()

    

    print('First three rows (after func())')
    print(data.head(3))
    new_path = os.path.join(save_to, 'changed_data.csv')
    try:
        data.to_csv(new_path)
        print("Saved to 'changed_data.csv'.")
        print('Time: %.6f' %(end-start))
    except:
        print('Something went wrong.')
    



if __name__=='__main__':
    input_file = 'q49_p7.75_up.csv'
    spec_column = 'Nu'
    from_content = 117.0917686
    to_content = np.nan
    cwd = os.getcwd()
    change_col_save_to_new_file(input_file, spec_column, from_content, to_content, cwd)


# 问题：法1法2哪个更好？简单从时间上看似乎是isin+at用时更短，是否有必要做区分？