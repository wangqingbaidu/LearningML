
# coding: utf-8
#通过检索学号直接定位到学生，将成绩输入到xls文件中
import xlrd                          
from xlutils.copy import copy    
name=input("输入文件名（如：学生信息.xls）:")   
workbook = xlrd.open_workbook(name)   ##读取表格以及复制表格准备修改
wb = copy(workbook)      
ws = wb.get_sheet(0)  
sheet = workbook.sheet_by_index(0)
ID_list=sheet.col_values(0)
num=input("第几次作业：")
num=7+int(num)
save=input("输入要保存的文件名（如：学生信息(新).xls）:")
while True:
    info=[]
    row=[]
    ID=input("输入学号后四位（输入'0'停止）：")
    if ID=="0":
        break
    for i in range(0,len(ID_list)):                     ##找学号位置
        if ID in ID_list[i]:
            info.append([sheet.cell(i,0).value,sheet.cell(i,1).value])
            row.append(i)
    if len(info)==0:
        print("未找到此学号")
        continue
    for j in info:
        print(j)
    if len(info)>1:                                  ##不同年级有学号后四位相同的，需要进行选择
        q=input("第一个还是第二个（请输入 1 或者 2）：")
        q=int(q)-1
        if q not in [0,1]:
            print("请重新输入")
            continue
        t=row[q]
    else:
        t=row[0]
    grade=input("输入成绩：")         
    grade=float(grade)
    while grade>10 or grade<7:                      ##检验成绩是否在正确范围
        if grade>=0 and grade<7:                             ##成绩在0-7之间，可能有误
            grade_new=input("成绩可能有误，请重新确认：")         
            grade_new=float(grade_new) 
            if grade==grade_new:                                 ##两次输入一致，成绩无误，跳出
                break
            else:                                                ##两次输入不一致，检验新值
                grade=grade_new
        else:                                                ##成绩小于0或大于10，一定有误
            grade=input("成绩有误，请重新输入：")                 ##输入新值 
            grade=float(grade)        
    ws.write(t,num,grade)                             ##写入成绩
    wb.save(save)




