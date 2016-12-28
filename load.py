# encoding:utf-8
# Python2 兼容
from __future__ import print_function, division
import tesnsorflow as tf
#from scipy.io import loadmat as load
#import matplotlib.pyplot as plt
#import numpy as np
import csv

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerater(f):
			pass
	return i + 1

filename = "data/forest_ag_int_1988_2014.csv"

#setup text reader
file_length = file_len(filename)
	
ifile = open('data/forest_ag_int_1988_2014.csv','rb')
reader = csv.reader(ifile)

rownum = 0
for row in reader:
    # Save header row.
    if rownum == 0:
        header = row
    else:
        colnum = 0
        for col in row:
            print '%-8s: %s' % (header[colnum], col)
            colnum += 1
            
    rownum += 1

ifile.close()