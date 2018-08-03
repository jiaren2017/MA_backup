
import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import regNetwork as regNN
import dnn_app_utils_v2 as dnn


#### street 1234
train_path = 'samples/1234/training_test_set/1234_with_block/';   
test_path  = 'samples/1234/training_test_set/1234_with_block/';  
model_path = "trained_model/2018_07_26/1234_with_block/"

####  training set for 1234 WITH block
train_Filename = ['testSet_mult_softmax.hdf5']  # testSet_mult_softmax.hdf5，  testSet_single_softmax.hdf5

#dataMat_org, dataMat_x_org, dataMat_y_org = read.read_hdf5_auto_v3(train_path, train_Filename, train_step_size = 1)
train_posSpeed_set, train_destination_set = read.read_hdf5_auto_v3(train_path, train_Filename, train_step_size = 2, mode = "softmax")