# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 18:56:45 2018

@author: zhouj_000

goal: prepare the training set

"""

import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import regNetwork as regNN
import random

############################################################################
#################     read traing and test data   ##########################
############################################################################

#path = 'samples/'
#Filename = ['mid-1_uni_PR10_PO10_F5_R10.hdf5','mid0_uni_PR10_PO10_F5_R10.hdf5','mid+1_uni_PR10_PO10_F5_R10.hdf5', 'mid+05_uni_PR10_PO10_F5_R10.hdf5', 'mid-05_uni_PR10_PO10_F5_R10.hdf5']


### street 1234
train_path = 'samples/'
train_Filename = ['test_112358.hdf5','test_random.hdf5']
dataMat_org, dataMat_x_org, dataMat_y_org = read.read_hdf5_auto_v2(train_path, train_Filename, train_step_size = 2)




