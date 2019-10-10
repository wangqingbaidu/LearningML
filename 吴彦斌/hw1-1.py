# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:11:00 2019

@author: WYB
"""
import pandas as pd

def func(input_file, spec_column, from_content, to_content, save_to):
    data=pd.read_csv(input_file)
    for i in data.index:
        if data.loc[i][spec_column]==from_content:
            data.loc[i][spec_column]=to_content
    data.to_csv(save_to,index=False,encoding="utf_8_sig")
    
func("q49_p7.75_up.csv", 'Nu', 10, 0, "q49_p7.75_up_new.csv")
