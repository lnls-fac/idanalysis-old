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
    data_col4 = []
    data_col5 = []
    begin_data = []
    gap_list = []
    for i,line in enumerate(my_file):
        if "Gap" in line:
            begin_data.append(i)
            gap = line[-3:-1]
            gap_list.append(int(gap))
    
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
                data_col4.append(float(list_data[3]))
                data_col5.append(float(list_data[4]))
            except ValueError:
                print('Error')
    my_file.close()

    return data_col1, data_col2, data_col3, data_col4, data_col5, nr_data, gap_list


def plot_charts(xpos_list, beff_list, bpeak_list, zpos_list, by_list, nr_data, gap_list,poles_proportion,filename,opmode):
    
    color_list = ['b','g','y','C1','r','C3','k']
    
    plt.figure(1)
    label_base = 'gap = '
    plt.xlabel('x position[mm]')
    plt.ylabel('Beff [T]')
    title = 'MAX IV IVU17 MODEL'
    # title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,gap in enumerate(gap_list):
        label = label_base + str(gap) + ' mm'
        plt.plot(xpos_list[i], beff_list[i], label=label, color=color_list[i])
    plt.grid()
    plt.legend()

    plt.figure(2)
    label_base = 'gap = '
    plt.xlabel('x position[mm]')
    plt.ylabel('Bpeak [T]')
    title = 'MAX IV IVU17 MODEL'
    # title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,gap in enumerate(gap_list):
        label = label_base + str(gap) + ' mm'
        plt.plot(xpos_list[i], bpeak_list[i], label=label, color=color_list[i])
    plt.grid()
    plt.legend()

    plt.figure(3)
    label_base = 'gap = '
    plt.xlabel('x position[mm]')
    plt.ylabel('Field Roll-off')
    title = 'MAX IV IVU17 MODEL'
    # title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,gap in enumerate(gap_list):
        label = label_base + str(gap) + ' mm'
        bpeak = np.array(bpeak_list[i])
        zero_idx_aux = np.where(np.array(xpos_list[i])>=0)
        zero_idx = zero_idx_aux[0][0]
        b0 = bpeak[zero_idx]
        roff = -1*(b0-bpeak)/b0
        plt.plot(xpos_list[i], roff, label=label, color=color_list[i])
    plt.xlim(-15,15)
    plt.ylim(-0.001,0.0006)
    plt.grid()
    plt.legend()

    plt.figure(4)
    label_base = 'gap = '
    plt.xlabel('z position[mm]')
    plt.ylabel('By[T]')
    title = 'MAX IV IVU17 MODEL'
    # title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,gap in enumerate(gap_list):
        label = label_base + str(gap) + ' mm'
        plt.plot(zpos_list[i], by_list[i], label=label, color=color_list[i])
    plt.grid()
    plt.legend()

    plt.show()
    
def run(filename):
    opmode = filename[0:3]
    rx, beff, bpeak, rz, by, nr_data, gap_list = readfile(filename)
    xpos_list = []
    beff_list = []
    bpeak_list = []
    zpos_list = []
    by_list = []
    for i,nr in enumerate(nr_data):
        if i == 0:
            xpos_list.append(rx[0:nr])
            beff_list.append(beff[0:nr])
            bpeak_list.append(bpeak[0:nr])
            zpos_list.append(rz[0:nr])
            by_list.append(by[0:nr])
        else:
            nr_data_split = nr_data[0:i]
            begin = np.sum(nr_data_split)
            xpos_list.append(rx[begin:begin+nr])
            beff_list.append(beff[begin:begin+nr])
            bpeak_list.append(bpeak[begin:begin+nr])
            zpos_list.append(rz[begin:begin+nr])
            by_list.append(by[begin:begin+nr])
    
    poles_proportion = filename[10:14]
    plot_charts(xpos_list, beff_list, bpeak_list, zpos_list, by_list, nr_data, gap_list,poles_proportion,filename,opmode)
if __name__ == "__main__":
    
    run("max_iv/ivu17.txt")

