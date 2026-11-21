# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:28:45 2024

@author: huanghe
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import tkinter as tk 
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog# 文件对话框，用于打开和保存文件
from PIL import Image,ImageTk# 图像处理
import webbrowser# 打开网页
import joblib# 模型加载和保存
import sys# 系统操作
import os# 操作系统接口
from lightgbm import LGBMRegressor as lgb
from PIL import Image
from tkinter import font
import lightgbm as lgb
import numpy as np  # 导入 numpy 并设置别名为 np


from diffusion_app.core import load_bundle, predict, read_table, INPUTS
model = load_bundle() 

## The layout design of the main interface
root= tk.Tk()
root.title("Predict diffusivity  of material on stacking")
root.configure(bg='#f7f6f2')
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 820, height =800, bg="#f7f6f2", highlightthickness=0) ## Main window size
canvas1.pack()

## The plate layout within the main interface
re1=canvas1.create_rectangle(50,20,770,505)
re2=canvas1.create_rectangle(50,525,770,725)
re3=canvas1.create_rectangle(70,100,380,370,outline='orange')
re4=canvas1.create_rectangle(400,100,750,370,outline='orange')
re4=canvas1.create_rectangle(70,395,750,485,outline='orange')
re4=canvas1.create_rectangle(70,615,750,705,outline='orange')
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 395, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 615, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',9),text='Author：He Huang ,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(580, 770, window=label_B)

## Message box (Related literature on molecular physical properties)
def cmx1():
    window = tk.Tk()     
    window.title('Warm prompt')     
    window.geometry('350x250')
    link = tk.Label(window, text='The physical properties of gases \nare known from the literature:\nhttps://doi.org/10.1039/B802426J'
                    , font=('microsoft yahei',10),anchor="center")
    link.place(x=30, y=50) 
    def open_url(event):
        webbrowser.open("https://doi.org/10.1039/B802426J", new=0)         
    link.bind("<Button-1>", open_url)    
btn1=tk.Button(root, text='Tooltip',font=('microsoft yahei',10), command=cmx1, bg='deeppink', fg='light cyan')
btn1.configure(bg='#ed8b32')
canvas1.create_window(450, 140, window=btn1)

##Message box (Instructions for Prediction of a single material diffusiivity)
def resize(w, h, w_box, h_box, pil_image): 
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height), Image.Resampling.LANCZOS)
       
def cmx2():
    messagebox.showinfo('Instructions for use', 'Enter material descriptors and gas properties.\nLongest 1D persistence is required.\nOutput: log10D and D in cm²/s.')

def cmx3():
    messagebox.showinfo('Batch prediction', 'Import CSV or XLSX with columns:\n' + ', '.join(INPUTS) + '\nDensity can be named ρ. Choose a save location after prediction.')

btn2=tk.Button(root,text='README',command=cmx2)
canvas1.create_window(120, 60,window=btn2)
btn3=tk.Button(root,text='README',command=cmx3)
canvas1.create_window(120, 575,window=btn3)

## Sets the label and entry for entering the nine descriptor 
label_Z = tk.Label(root,font=('microsoft yahei',13),text='Predict diffusivity of material')
canvas1.create_window(415, 20, window=label_Z)

label_L = tk.Label(root,font=('microsoft yahei',11),text='Physical property of material')
canvas1.create_window(225, 100, window=label_L)

label1 = tk.Label(root,font=('microsoft yahei',10),text='φ：') ## create 1st label box 
canvas1.create_window(160, 140, window=label1)
entry1 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 1st entry box 
canvas1.create_window(300, 140, window=entry1)

label2 = tk.Label(root,font=('microsoft yahei',10), text='PLD (Å): ') ## create 2st label box 
canvas1.create_window(160, 180, window=label2)
entry2 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 2nd entry box
canvas1.create_window(300, 180, window=entry2)

label3 = tk.Label(root,font=('microsoft yahei',10), text='LCD (Å): ') ## create 3st label box 
canvas1.create_window(160, 220, window=label3)
entry3 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 3nd entry box
canvas1.create_window(300, 220, window=entry3)

label4 = tk.Label(root,font=('microsoft yahei',10,"italic"), text='ρ') ## create 4st label box 
canvas1.create_window(145, 260, window=label4)
l0 = tk.Label(root,font=('microsoft yahei',10), text='(kg/m^3): ') 
canvas1.create_window(190, 260, window=l0)
entry4 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 4nd entry box
canvas1.create_window(300, 260, window=entry4)

label5 = tk.Label(root, font=('microsoft yahei',10),text='VSA (m^2/cm^3): ') ## create 5st label box 
canvas1.create_window(160, 300, window=label5) 
entry5 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 5nd entry box
canvas1.create_window(300, 300, window=entry5)

label_R = tk.Label(root,font=('microsoft yahei',11),text='Physical property of gas molecules') 
canvas1.create_window(575, 100, window=label_R)

label6 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dia') ## create 1st 6abel box 
canvas1.create_window(490, 180, window=label6)
l1 = tk.Label(root, font=('microsoft yahei',10),text='(Å):')
canvas1.create_window(525, 180, window=l1) 
entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 6nd entry box
canvas1.create_window(660, 180, window=entry6)

label7 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pol') ## create 7st label box 
canvas1.create_window(430, 220, window=label7)
l2 = tk.Label(root, font=('microsoft yahei',10,),text='(×10^25 cm^3): ')
canvas1.create_window(520, 220, window=l2)
entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 7nd entry box
canvas1.create_window(660, 220, window=entry7)

label8 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dip') ## create 8st label box 
canvas1.create_window(430, 260, window=label8) 
l3 = tk.Label(root, font=('microsoft yahei',10),text='(×10^18 cm): ')
canvas1.create_window(510, 260, window=l3)  
entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 8nd entry box
canvas1.create_window(660, 260, window=entry8)

label9 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='M') ## create 9st label box 
canvas1.create_window(430, 300, window=label9)
l4 = tk.Label(root, font=('microsoft yahei',10),text='(g/mol): ')
canvas1.create_window(525, 300, window=l4) 
entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660, 300, window=entry9)

## The linkage between the physical properties of molecules and molecules is realized
## Sets four properties: 1-dia,2-pol,3-Dip,4-Qua
input1 = 0
input2 = 0
input3 = 0
input4 = 0
def calc(event=None):
    gas=cm1.get()
    if gas not in model['gas_presets']: return
    props=model['gas_presets'][gas]
    for entry,key in zip([entry6,entry7,entry8,entry9],['Dia','Pol','Dip','M']):
        entry.delete(0,'end');entry.insert(0,str(props[key]))

## Create a Drop-down box
var1 = tk.StringVar() ## Create a variable 
cm1 = ttk.Combobox(root, textvariable=var1,font=('microsoft yahei',10)) ## Create a drop-down menu
cm1["value"] = ("C1", "C2", "C3","C4","C5","C6","C7") ## The contents of a drop-down menu
canvas1.create_window(620, 140, window=cm1)
cm1.bind('<<ComboboxSelected>>', calc) ## Binding 'calc' events
  
## Main interface for input of nine descriptor values (a single molecule diffusivity)
result_var=tk.StringVar(value='—')
def values():
    try:
        entries=[entry1,entry2,entry3,entry4,entry5,entry6,entry7,entry8,entry9,entry10]
        frame=pd.DataFrame([{key:entry.get() for key,entry in zip(INPUTS,entries)}])
        row=predict(frame,model).iloc[0]
        result_var.set(f"D = {row.predicted_D_cm2_s:.4e} cm²/s   |   log10D = {row.predicted_log10_D:.5f}")
    except Exception as error:
        messagebox.showerror('Prediction error',str(error))
result_label=tk.Label(root,textvariable=result_var,font=('microsoft yahei',11),wraplength=490)
canvas1.create_window(470, 435,window=result_label)

## button to call the 'values' command above       
button1 = tk.Button (root,font=('microsoft yahei',10), text='Predicted D',command=values, bg='deeppink', fg='white')
button1.configure(bg='#ed8b32') 
canvas1.create_window(130, 435, window=button1)

## Batch prediction of material diffusivity
label_Z1 = tk.Label(root,font=('microsoft yahei',12),text='Batch prediction of material diffusivity')
canvas1.create_window(415, 525, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open exce')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)
 
button_import = tk.Button(root, text="Import File",font=('microsoft yahei',10),command=open_file)
button_import.configure(bg='#ed8b32') 
canvas1.create_window(280, 575, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('microsoft yahei',10),width=30)
canvas1.create_window(520, 575, window=entry_filename)

## Output LGBM model prediction results
batch_var=tk.StringVar(value='')
def print_file():
    try:
        src=entry_filename.get()
        if not src: return messagebox.showinfo('Import file','Choose an input file first.')
        result=predict(read_table(src),model)
        dest=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='Batch_Predicted_D.csv',filetypes=[('CSV','*.csv'),('Excel','*.xlsx')])
        if not dest: return
        if dest.lower().endswith('.csv'):result.to_csv(dest,index=False,encoding='utf-8-sig')
        else:result.to_excel(dest,index=False)
        batch_var.set(f'Saved {len(result)} predictions.');messagebox.showinfo('Prediction complete',dest)
    except Exception as error:
        messagebox.showerror('Batch prediction error',str(error))
batch_label=tk.Label(root,textvariable=batch_var,font=('microsoft yahei',10))
canvas1.create_window(450, 655,window=batch_label)

## Prediction button
but_pre=tk.Button(root,font=('microsoft yahei',10)
             , text='Batch Predicted D', bg='deeppink', fg='white', command=print_file)
canvas1.create_window(160, 655, window=but_pre)

label10=tk.Label(root,font=('microsoft yahei',9),text='Longest 1D persistence:')
canvas1.create_window(170,340,window=label10)
entry10=tk.Entry(root,font=('microsoft yahei',10),width=12,justify='center')
canvas1.create_window(300,340,window=entry10)
cm1.current(0);calc()
for widget in root.winfo_children():
    if isinstance(widget,tk.Button):
        widget.configure(bg='#ed8b32',fg='white',activebackground='#d8751e',activeforeground='white',relief='flat',bd=0,padx=12,pady=6,cursor='hand2',font=('Microsoft YaHei UI',10))
    elif type(widget) is tk.Entry:
        widget.configure(bg='white',fg='#243247',relief='solid',bd=1,insertbackground='#ed8b32')
    elif isinstance(widget,tk.Label):
        widget.configure(bg='#f7f6f2',fg='#243247')
root.mainloop() 

