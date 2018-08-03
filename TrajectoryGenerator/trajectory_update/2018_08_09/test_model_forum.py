import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import neural_network as NN
import random



#### parameters
train_path_forum = 'samples/forum/training_test_set/' 
test_path_forum  = 'samples/forum/training_test_set/'
model_path_forum = "trained_model/2018_08_09/forum/"
last_steps = "_L1"
file_format = ".dat"
step_size = 1
machine_mode = "classification"
date = "_04Jan"       
FMT = "_S1"          ## F: full, S: select


############################################################################
###################  load training and test set   #########################
############################################################################


###  load data set for regression mode

training_set_xy_file = 'training_set_xy_forum' + date + FMT + last_steps + file_format
training_set_x_file = 'training_set_x_forum' + date + FMT + last_steps + file_format
training_set_y_file = 'training_set_y_forum' + date + FMT + last_steps + file_format
test_set_x_filename = "test_set_x_1234Block" + last_steps + file_format
test_set_y_filename = "test_set_y_1234Block" + last_steps + file_format

dataMat_org = read.load_data(train_path_forum + training_set_xy_file)
dataMat_x_org = read.load_data(train_path_forum + training_set_x_file)
dataMat_y_org = read.load_data(train_path_forum + training_set_y_file)
#dataMat_x_test = read.load_data(test_path_forum + test_set_x_filename)
#dataMat_y_test = read.load_data(test_path_forum + test_set_y_filename)



###  load data set for classification mode
training_posSpeed_set_file = 'training_posSpeed_set_forum' + date + FMT + last_steps + file_format
train_destination_set_file = 'training_destination_set_forum'  + date + FMT + last_steps + file_format
training_posSpeed_set = read.load_data(train_path_forum + training_posSpeed_set_file)
train_destination_set = read.load_data(train_path_forum + train_destination_set_file)

'''
test_posSpeed_set_file = 'test_1234Block_posSpeed_set' + last_steps + file_format
test_destination_set_file = 'test_1234Block_destination_set' + last_steps + file_format
test_posSpeed_set = read.load_data(test_path + test_posSpeed_set_file)
test_destination_set = read.load_data(test_path + test_destination_set_file)
'''

############################################################################
#####################   load neural network model   ################## 
############################################################################

#####  classification mode  #####
NN_forum_class_model_file = 'regNN_forum_classModel_trained' + date + FMT + last_steps + file_format
NN_forum_class_model_trained = read.load_data(model_path_forum + NN_forum_class_model_file)

## normalizing training data
norm_parameters_x, train_x = NN.normData(training_posSpeed_set)
train_y_destination = train_destination_set.T


############################################################################
####################    evaluate nerual network model      #################
############################################################################

#############   test classification   #############
## evaluate with train_x ----------------------------------
print('\n\n---------------------   evaluate neural network model for classification   ------------------------------------')

train_destination_predNN = NN.network_predict_v3(train_x, NN_forum_class_model_trained, prediction_mode = "classification", foreAct = 'relu')
train_destination_prediction, train_error_list = NN.prediction_destination(train_destination_predNN, train_y_destination)
print("\ncorrect recognition rate (data: train, target: destination, mode: NN): ", train_destination_prediction)    #  dataMat_x[:,-1]: transfer the acceleration !

'''
test_destination_predNN = NN.network_predict_v3(test_posSpeed_set.T, NN_1234_class_model_trained, norm_parameters_NN = norm_parameters_x, prediction_mode = "classification", foreAct = 'relu')
test_destination_prediction, test_error_list = NN.prediction_destination(test_destination_predNN, test_destination_set.T)
print("\ncorrect recognition rate (data: test, target: destination, mode: NN): ", test_destination_prediction)    #  dataMat_x[:,-1]: transfer the acceleration !
'''

#plot.plot_forum(dataMat_org[::4,:], dataMat_error=training_posSpeed_set[train_error_list,:],  origShow=False, predShow =True, minX_plot = 0, maxX_plot = 640, minY_plot = 0, maxY_plot = 480)












