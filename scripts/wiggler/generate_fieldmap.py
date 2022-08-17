#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from imaids import utils as utils


FOLDER_BASE = '/home/ximenes/repos-dev/ids-data/Wiggler/'


def readfield(file_name, init):
    end_flag = 0
    with open(file_name, encoding='utf-8', errors='ignore') as my_file:
        data_col1 = []
        data_col2 = []
        col1_values = []
        col2_values = []
        for i,line in enumerate(my_file):
            if i >= init:
                list_data = line.split('\t') #returns a list
                if 'M' not in list_data[0]:
                    if end_flag == 0:
                        try:
                            data_col1.append(float(list_data[3]))
                            data_col2.append(float(list_data[5]))
                        except ValueError:
                            data_col1.append((list_data[3]))
                            data_col2.append((list_data[5]))
                else:
                    end_flag = 1
    my_file.close()
    z = np.array(data_col1)
    B = np.array(data_col2)
    return z,B


def readfile_axis(x):
    
    folder = FOLDER_BASE + 'gap 059.60 mm/ponta hall/mapeamento/'
    fieldname = "Map2701_X=" + str(x) + ".dat"
    filename = folder + fieldname
    rz_file,By_file = readfield(filename, 24) # 24 14669
    By_file = By_file/10e3

    By = []
    for z in zvalues:
        difference_array = np.absolute(rz_file - z)
        index = difference_array.argmin()
        By.append(By_file[index])
        
    return By,fieldname


def run(x_list):

    By_dict = dict()
    for x in x_list:
        By,_ = readfile_axis(x=x)
        By_dict[x] = By[0:2]
    
    yvalues = np.arange(-1,2,1)
        

if __name__ == "__main__":
    
    zmin = -419
    zmax = 3678
    zvalues = np.arange(zmin,zmax+1,1)
    x_list = [-20, -10, -5, -1, 0, 1, 5, 10, 20]
    run(x_list=x_list)

    # plt.plot(zvalues,By, color='b')
    # plt.grid(True)
    # plt.title("Vertical Field for " + fieldname[int(len(fieldname)/2):-4])
    # plt.xlabel('Longitudinal distance [mm]')
    # plt.ylabel('B [T]')
    # plt.show()
