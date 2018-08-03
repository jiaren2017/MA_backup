import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import neural_network as NN
import dnn_app_utils_v2 as dnn


#### forum
train_path_forum = 'samples/forum/training_test_set/' 
test_path_forum  = 'samples/forum/training_test_set/'
model_path_forum = "trained_model/2018_08_09/forum/"
last_steps = "_L1"
file_format = ".dat"
step_size = 1
machine_mode = "classification"
date = "_12Sep"       
FMT = "_S1"          ## F: full, S: select,  S1: forum_min_step_size = 5



###############################   read forum trajectory data   ######################################

path_raw_data = "samples/forum/"
filename = "forum_trajectory_data.txt"
forum = ["tracks.24Aug.txt" ,"tracks.25Aug.txt","tracks.26Aug.txt","tracks.27Aug.txt","tracks.28Aug.txt",
         "tracks.01Sep.txt","tracks.02Sep.txt","tracks.04Sep.txt","tracks.05Sep.txt","tracks.06Sep.txt",
         "tracks.10Sep.txt","tracks.11Sep.txt","tracks.12Sep.txt","tracks.04Jan.txt"
         ]
forum_1 = ["tracks.12Sep.txt"]
forum_test = ["tracks.10Sep.txt","tracks.11Sep.txt","tracks.12Sep.txt"]

## Distance in (mm/s) = pixel/frame * 24.7mm/pixel * 9 frame/s    each second 10 frames
forum_min_step_size = 5     ## min. step size in each direction: x(t) - x(t-1)      # S1:forum_min_step_size = 5
forum_max_step_size = 100    ## max. step size: sqrt(square(x(t)-x(t-1)) + square(y(t)-y(t-1)))
forum_mean_step_size = 7    ## mean distance in pixel



############################################################################
#### generate the destination set
############################################################################
destination_set = read.build_destination_set(path_raw_data, forum, last_steps = 1)




############################################################################
#### produce and save trajectory Training set
############################################################################

############   regression mode    ############

'''
dataMat_org, dataMat_x_org, dataMat_y_org, removed_steps_list_reg,  removed_lines_list_reg, destination_set = read.load_forum_trajectory_set(path_raw_data, forum_test, train_step_size = step_size, mode = "regression", minStepSize = forum_min_step_size, maxStepSize = forum_max_step_size)
##  save training set into file
training_set_xy_file = 'training_set_xy_forum' + date + FMT + last_steps + file_format
training_set_x_file = 'training_set_x_forum' + date + FMT + last_steps + file_format
training_set_y_file = 'training_set_y_forum' + date + FMT + last_steps + file_format
read.save_data(dataMat_org, train_path_forum + training_set_xy_file)
read.save_data(dataMat_x_org, train_path_forum + training_set_x_file)
read.save_data(dataMat_y_org, train_path_forum + training_set_y_file)
'''

############   classification mode   ############

'''
train_posSpeed_set, train_destination_set, removed_steps_list_class,  removed_lines_list_class = read.load_forum_trajectory_set(path_raw_data, forum_1, train_step_size = step_size, mode = "classification", minStepSize = forum_min_step_size, maxStepSize = forum_max_step_size)
##  save training set into file
training_posSpeed_set_file = 'training_posSpeed_set_forum' + date + FMT + last_steps + file_format
train_destination_set_file = 'training_destination_set_forum'  + date + FMT + last_steps + file_format
read.save_data(train_posSpeed_set, train_path_forum + training_posSpeed_set_file)
read.save_data(train_destination_set, train_path_forum + train_destination_set_file)
'''

### check the dataSet
# plot.plot_forum(dataMat_org[:,:], origShow=True, predShow =False, minX_plot = 0, maxX_plot = 640, minY_plot = 0, maxY_plot = 480)


############################################################################
###################  load training and test set   #########################
############################################################################
### forum

'''
training_set_xy_file = 'training_set_xy_forum' + date + FMT + last_steps + file_format
training_set_x_file = 'training_set_x_forum' + date + FMT + last_steps + file_format
training_set_y_file = 'training_set_y_forum' + date + FMT + last_steps + file_format
#test_set_x_filename = "test_set_x_1234Block" + last_steps + file_format
#test_set_y_filename = "test_set_y_1234Block" + last_steps + file_format  

dataMat_org = read.load_data(train_path_forum + training_set_xy_file)
dataMat_x_org = read.load_data(train_path_forum + training_set_x_file)
dataMat_y_org = read.load_data(train_path_forum + training_set_y_file)
#dataMat_x_test = read.load_data(test_path + test_set_x_filename)
#dataMat_y_test = read.load_data(test_path + test_set_y_filename)



training_posSpeed_set_file = 'training_posSpeed_set_forum' + date + FMT + last_steps + file_format
train_destination_set_file = 'training_destination_set_forum'  + date + FMT + last_steps + file_format
train_posSpeed_set = read.load_data(train_path_forum + training_posSpeed_set_file)
train_destination_set = read.load_data(train_path_forum + train_destination_set_file)
'''

############################################################################
####################    neural network       ###############################
############################################################################

##############################################
###  train trajectory with neural network  ###
##############################################

##########  build the regression model  ##########

'''
#### normalize training set
norm_parameters_x, train_x = NN.normData(dataMat_x_org[:,:-1])
norm_parameters_y, train_y = NN.normData(dataMat_y_org[:,:-1])
train_x_target = dataMat_x_org[:,-1].T
train_y_target = dataMat_y_org[:,-1].T


#### hidden layers
layers_dims = [len(train_x), 64, 32, 16, 1]  # version: 1


#### build  model
parameters_x = NN.L_layer_model(train_x, train_x_target, layers_dims, optimizer = "adam", enable_mini_batch = True, 
                                   learning_rate = 0.01, num_iterations = 1000, print_cost = True)  


parameters_y = NN.L_layer_model(train_y, train_y_target, layers_dims, optimizer = "adam", enable_mini_batch = True, 
                                   learning_rate = 0.01, num_iterations = 1000, print_cost = True)   


#### save neural network into file
regNN_1234_x_file = 'regNN_1234Block_x_trained_v1' + last_steps + file_format
regNN_1234_y_file = 'regNN_1234Block_y_trained_v1' + last_steps + file_format
NN.saveNetwork(parameters_x, model_path + regNN_1234_x_file)
NN.saveNetwork(parameters_y, model_path + regNN_1234_y_file)

'''


##########  build classification model  ##########

'''
#### normalize training set
norm_parameters_x, train_x = NN.normData(train_posSpeed_set)
train_y_destination = train_destination_set.T

#### hidden layers
layers_dims = [len(train_x), 20, 9]  # version: 1

#### build  model
class_model_trained = NN.L_layer_model(train_x, train_y_destination, layers_dims, learning_mode = "classification", optimizer = "adam", 
                                          enable_mini_batch = True,  learning_rate = 0.001, num_iterations = 500, print_cost = True)  

#### save neural network into file
regNN_forum_model_file = 'regNN_forum_classModel_trained' + date + FMT + last_steps + file_format
read.save_data(class_model_trained, model_path_forum + regNN_forum_model_file)
'''
