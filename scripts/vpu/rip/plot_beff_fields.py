#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt
import imaids.utils as utils

def readfile(file_name):

    my_file = open(file_name)
    data_col1 = []
    data_col2 = []
    data_col3 = []
    begin_data = []
    width_list = []
    for i,line in enumerate(my_file):
        if "Block width" in line:
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

def plot_k(heights_list, beff_list, roff_list, nr_data, width_list,period,poles_proportion):
    plt.figure(1)
    label_base = 'Block width = '
    plt.xlabel('Block height [mm]')
    plt.ylabel('K')
    title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,width in enumerate(width_list):
        label = label_base + str(width) + ' mm'
        beff = _np.array(beff_list[i])
        k = utils.calc_deflection_parameter(b_amp=beff, period_length=period*1e-3)
        plt.plot(heights_list[i], k, label=label)
    plt.grid()
    plt.legend()

    plt.figure(2)
    label_base = 'Block width = '
    plt.xlabel('Block height [mm]')
    plt.ylabel('Field roll-off [%]')
    title = "Poles proportion = " + poles_proportion + "%"
    plt.title(title)
    for i,width in enumerate(width_list):
        label = label_base + str(width) + ' mm'
        roff = _np.array(roff_list[i])
        plt.plot(heights_list[i], roff, label=label)
    plt.grid()
    plt.show()
    
def run(filename,period):
    heights, beffs, roffs, nr_data, width_list = readfile(filename)
    heights_list = []
    beff_list = []
    roff_list = []
    for i,nr in enumerate(nr_data):
        if i == 0:
            height = heights[0:nr]    
            heights_list.append(height)
            beff_list.append(beffs[0:nr])
            roff_list.append(roffs[0:nr])
        else:
            nr_data_split = nr_data[0:i]
            begin = _np.sum(nr_data_split)
            height = heights[begin:begin+nr]
            heights_list.append(height)
            beff_list.append(beffs[begin:begin+nr])
            roff_list.append(roffs[begin:begin+nr])
    
    poles_proportion = filename[5:9]
    plot_k(heights_list, beff_list, roff_list, nr_data, width_list,period,poles_proportion)
if __name__ == "__main__":
    
    run("Beff_70.0%.txt",29)

