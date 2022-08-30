#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt
from scipy import optimize as optimize

def readfile(file_name):
    
    my_file = open(file_name)
    data_col1 = []
    data_col2 = []
    data_col3 = []
    begin_data = []
    width_list = []
    for i,line in enumerate(my_file):
        if "Blocks width" in line:
            begin_data.append(i)
            width = line[-3:-1]
            width_list.append(int(width))
    
    begin_data.append(i+1)
    begin = _np.array(begin_data)
    begin += 2
    nr_data = _np.diff(begin) - 2
    
    valid_lines = []
    for i in _np.arange(len(begin)-1):
        nr_line = nr_data[i] -1
        valid_lines.append(begin[i])
        for j in _np.arange(nr_line):
            valid_lines.append(begin[i]+j+1)
    my_file.close()
    
    my_file = open(file_name)
    for i, line in enumerate(my_file):
        if i in valid_lines:
            list_data = line.split('\t') #returns a list
            try:
                data_col1.append(float(list_data[0]))
                data_col2.append(float(list_data[1]))
                data_col3.append(float(list_data[2]))
            except ValueError:
                data_col1.append((list_data[0]))
                data_col2.append((list_data[1]))
                data_col3.append((list_data[2]))
    my_file.close()

    return data_col1, data_col2, data_col3, nr_data, width_list

def plot_length_height(lengths_list, height_list, width_list):
    x = _np.linspace(0.4,3,100)
    label_base = 'Block width = '
    color_list = ['b','g','y','r','k']
    plt.figure(1)
    plt.xlabel('ID length [m]')
    plt.ylabel('Block height [mm]')
    plt.title('Minimum necessary block height for K=2.15')
    for i,width in enumerate(width_list):
        label = label_base + str(width) + ' mm'
        plt.plot(lengths_list[i], height_list[i], 'o', label=label, color=color_list[i])
        coef = fit_curve(lengths_list[i],height_list[i])
        y = calc_function(x, coef[0],coef[1],coef[2],coef[3],coef[4])
        print(coef)
        plt.plot(x,y,color_list[i])
    
    #plt.ylim(35,120)
    #plt.xlim(0.4,2)
    plt.grid()
    plt.legend()
    plt.show()

def plot_length_roff(lengths_list, roll_off_list, width_list):
    x = _np.linspace(0.4,2,100)
    label_base = 'Block width = '
    color_list = ['b','g','y','r','k']
    plt.figure(1)
    plt.xlabel('ID length [m]')
    plt.ylabel('Field roll-off [%]')
    plt.title('Field roll-off at X=10mm for K=2.15')
    for i,width in enumerate(width_list):
        label = label_base + str(width) + ' mm'
        plt.plot(lengths_list[i], _np.abs(roll_off_list[i]),linestyle='--',marker='o', label=label, color=color_list[i])
        
    #plt.ylim(0,0.5)
    plt.grid()
    plt.legend()
    plt.show()

def calc_function(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def fit_curve(lengths, height):
    a0 = 2
    b0 = -3
    c0 = 1
    d0 = 1
    e0 = 40
    
    opt = optimize.curve_fit(
        calc_function, lengths, height, p0=(a0, b0, c0, d0, e0))[0]

    return opt

def run(filename):
    lengths, heights, roll_offs, nr_data, width_list = readfile(filename)
    lengths_list = []
    height_list = []
    roll_off_list = []
    for i,nr in enumerate(nr_data):
        if i == 0:
            length = lengths[0:nr]    
            lengths_list.append(length)
            height_list.append(heights[0:nr])
            roll_off_list.append(roll_offs[0:nr])
        else:
            nr_data_split = nr_data[0:i]
            begin = _np.sum(nr_data_split)
            length = lengths[begin:begin+nr]
            lengths_list.append(length)
            height_list.append(heights[begin:begin+nr])
            roll_off_list.append(roll_offs[begin:begin+nr])
    
   
    plot_length_height(lengths_list, height_list, width_list)
    plot_length_roff(lengths_list, roll_off_list, width_list)
if __name__ == "__main__":
    
    run("IDs_length35mm.txt")
    
    
    
    

