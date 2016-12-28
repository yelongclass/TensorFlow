# encoding:utf-8
# Python2 兼容
#from __future__ import print_function, division
#import tensorflow as tf
#from scipy.io import loadmat as load
#import matplotlib.pyplot as plt
import csv
from itertools import islice
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#Read dataset
in_file = 'data/forest_ag_int_1988_2014.csv'
with open(in_file,'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#Print  whole row from begin
	for i in range(2):
		print spamreader.next()

#There are total 1376867 row in the dataset.
#	row_count = sum(1 for row in spamreader)  # fileObject is your csv.reader
#	print  ('There are total %s row in the dataset.' % row_count)





#outcomes = reader['f1988tr2']
#data = reader.drop('f1988tr2', axis = 1)
#print(data.head())

#Print the whole dataset
'''
rownum = 0
for row in reader:
    # Save header row.
    if rownum == 0:
        header = row
    else:
        colnum = 0
        for col in row:
            print ('%-8s: %s' % (header[colnum], col))
            colnum += 1
    rownum += 1
ifile.close()
'''
