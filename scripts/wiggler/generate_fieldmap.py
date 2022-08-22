#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

FOLDER_BASE = '/home/gabriel/repos-dev/ids-data/Wiggler/'

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
    rz_file,By_file = readfield(filename,24)#24 14669
    By_file = By_file/10e3

    By = []
    for z in zvalues:
        difference_array = np.absolute(rz_file - z)
        index = difference_array.argmin()
        By.append(By_file[index])
        
    return By,fieldname

def run(xvalues,ymax,ystep):
    
    By_dict = dict()
    for x in xvalues:
        By,_ = readfile_axis(x=x)
        By_dict[x] = By

    yvalues = np.arange(-ymax,ymax+ystep,ystep)
    
    x_col1 = np.ones(len(xvalues)*len(yvalues)*len(zvalues))
    y_col2 = np.ones(len(xvalues)*len(yvalues)*len(zvalues))
    z_col3 = np.ones(len(xvalues)*len(yvalues)*len(zvalues))
    b_col4 = np.ones(len(xvalues)*len(yvalues)*len(zvalues))
    b_col5 = np.ones(len(xvalues)*len(yvalues)*len(zvalues))
    line = 0
    for i,z in enumerate(zvalues):
        for y in yvalues:
            for x in xvalues:
                x_col1[line] = x
                y_col2[line] = y
                z_col3[line] = z
                b_col4[line] = By_dict[x][i]
                b_col5[line] = 0
                line+=1

    my_file = open("Fieldmap.fld","w") #w=writing
    my_file.write('X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n')
    my_file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    for i in range(x_col1.size):
        my_file.write("{:.1f}\t{:.1f}\t{:.1f}\t{:.5e}\t{:.5e}\t{:.5e}\n".format(x_col1[i],y_col2[i],z_col3[i],b_col5[i],b_col4[i],b_col5[i]))
    my_file.close()

if __name__ == "__main__":
    
    zmin = -419
    zmax = 3678
    zvalues = np.arange(zmin,zmax+1,1)
    xvalues = [-45,-35,-25,-20,-16,-12,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,12,16,20,25,35]
    run(xvalues=xvalues,ymax=12,ystep=1)

