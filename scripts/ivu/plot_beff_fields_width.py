#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
import imaids.utils as utils

from scipy import optimize as optimize

def readfile(file_name):

    my_file = open(file_name)
    data_col1 = []
    data_col2 = []
    data_col3 = []
    begin_data = []
    height_list = []
    for i,line in enumerate(my_file):
        if "Block height" in line:
            begin_data.append(i)
            height = line[-3:-1]
            height_list.append(int(height))
    
    begin_data.append(i+1)
    begin = np.array(begin_data)
    begin += 2
    nr_data = np.diff(begin) - 2
    
    valid_lines = []
    for i in np.arange(len(begin)-1):
        nr_line = nr_data[i] -1
        valid_lines.append(begin[i])
        for j in np.arange(nr_line):
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

    return data_col1, data_col2, data_col3, nr_data, height_list

def calc_function(x, a, b, c):
    return a*np.exp(-1*b*(x-18)) + c

def fit_curve(widths, rollofs):
    # a0 = 2
    # b0 = -3
    # c0 = 1
   
    
    opt = optimize.curve_fit(
        calc_function, widths, rollofs)[0]

    return opt

def plot_k(widths_list, beff_list, roff_list, nr_data, height_list,period,poles_proportion,filename):
    color_list = ['b','g','y','C1','r','C3','k']
    plt.figure(1)
    label_base = 'Block height = '
    plt.xlabel('Block width [mm]')
    plt.ylabel('K')
    title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,height in enumerate(height_list):
        label = label_base + str(height) + ' mm'
        beff = np.array(beff_list[i])
        k = utils.calc_deflection_parameter(b_amp=beff, period_length=period*1e-3)
        plt.plot(widths_list[i], k, label=label, color=color_list[i])
    plt.grid()
    plt.legend()
    plt.savefig(filename[0:14] + '/plot_K.png',dpi=300)

    plt.figure(2)
    label_base = 'Block height = '
    plt.xlabel('Block width [mm]')
    plt.ylabel('Field roll-off [%]')
    title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    width = np.arange(20,60,0.5)
    
    for i,height in enumerate(height_list):
        label = label_base + str(height) + ' mm'
        roff = np.array(roff_list[i])
        param = fit_curve(widths_list[i],roff)
        rollfit = calc_function(width,param[0],param[1],param[2])
        plt.plot(widths_list[i], roff, '.', label=label, color=color_list[i])
        plt.plot(width, rollfit, '--', color=color_list[i])
    plt.grid()
    plt.legend()
    plt.xlim(30,60)
    plt.ylim(-0.5,0.3)
    plt.savefig(filename[0:14] + '/plot_rolloff.png',dpi=300)
    plt.show()
    
def run(filename):
    opmode = filename[0:3]
    opconfig = dict([
        ('op1',[17.7,4.2,1.34]),
        ('op2',[17.5,3.75,1.24]),
        ('op3',[18.5,4.2,1.24])])
    period = opconfig[opmode][0]
    print(period)
    widths, beffs, roffs, nr_data, height_list = readfile(filename)
    widths_list = []
    beff_list = []
    roff_list = []
    for i,nr in enumerate(nr_data):
        if i == 0:
            width = widths[0:nr]    
            widths_list.append(width)
            beff_list.append(beffs[0:nr])
            roff_list.append(roffs[0:nr])
        else:
            nr_data_split = nr_data[0:i]
            begin = np.sum(nr_data_split)
            width = widths[begin:begin+nr]
            widths_list.append(width)
            beff_list.append(beffs[begin:begin+nr])
            roff_list.append(roffs[begin:begin+nr])
    
    poles_proportion = filename[10:14]
    plot_k(widths_list, beff_list, roff_list, nr_data, height_list,period,poles_proportion,filename)


if __name__ == "__main__":
    run("op3_width_90.0/Beff_op3_width_90.0%.txt")

