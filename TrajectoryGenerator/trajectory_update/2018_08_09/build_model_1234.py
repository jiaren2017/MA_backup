import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import neural_network as NN
import dnn_app_utils_v2 as dnn
###############################   read data   ######################################
### average walk speed is 1.2 m/s.  so the distance will be 6 meters in 5 seconds. 
### the scale of simulation is: 6 meters is equal to 100 pixels, which in turn gives the speed with equation:  speed(m/s) = pixels in speed / 16.6

############################################################################
###################  generate training and test set   #########################
############################################################################

#### street 1234
train_path_1234 = 'samples/1234/training_test_set/1234_with_block/';   
test_path_1234  = 'samples/1234/training_test_set/1234_with_block/';  
model_path_1234 = "trained_model/2018_08_09/1234_with_block/"
last_steps = "_L1"
file_format = ".dat"
step_size = 1
machine_mode = "classification"




####  training set for 1234 WITH block
train_Filename = ['1_2_PR1_PO05_QUAD_F3_R10.hdf5', '1_2_PR2_PO05_QUAD_F3_R10.hdf5', '1_2_PR3_PO05_QUAD_F3_R10.hdf5', 
                  '1_3_PR1_PO05_QUAD_F3_R10.hdf5', '1_3_PR2_PO05_QUAD_F3_R10.hdf5', '1_3_PR3_PO05_QUAD_F3_R10.hdf5', '1_3_PR4_PO05_QUAD_F3_R10.hdf5',
                  '1_4_PR2_PO05_QUAD_F3_R10.hdf5', '1_4_PR3_PO05_QUAD_F3_R10.hdf5', '1_4_PR4_PO05_QUAD_F3_R10.hdf5',
                  
                  '2_1_PR1_PO05_QUAD_F3_R10.hdf5', '2_1_PR2_PO05_QUAD_F3_R10.hdf5', '2_1_PR3_PO05_QUAD_F3_R10.hdf5', 
                  '2_3_PR2_PO05_QUAD_F3_R10.hdf5', '2_3_PR3_PO05_QUAD_F3_R10.hdf5', '2_3_PR4_PO05_QUAD_F3_R10.hdf5',
                  '2_4_PR1_PO05_QUAD_F3_R10.hdf5', '2_4_PR2_PO05_QUAD_F3_R10.hdf5', '2_4_PR3_PO05_QUAD_F3_R10.hdf5', '2_4_PR4_PO05_QUAD_F3_R10.hdf5',
                  
                  '3_1_PR1_PO05_QUAD_F3_R10.hdf5', '3_1_PR2_PO05_QUAD_F3_R10.hdf5', '3_1_PR3_PO05_QUAD_F3_R10.hdf5', '3_1_PR4_PO05_QUAD_F3_R10.hdf5',
                  '3_2_PR2_PO05_QUAD_F3_R10.hdf5', '3_2_PR3_PO05_QUAD_F3_R10.hdf5', '3_2_PR4_PO05_QUAD_F3_R10.hdf5',
                  '3_4_PR1_PO05_QUAD_F3_R10.hdf5', '3_4_PR2_PO05_QUAD_F3_R10.hdf5', '3_4_PR3_PO05_QUAD_F3_R10.hdf5',
                  
                  '4_1_PR2_PO05_QUAD_F3_R10.hdf5', '4_1_PR3_PO05_QUAD_F3_R10.hdf5', '4_1_PR4_PO05_QUAD_F3_R10.hdf5',
                  '4_2_PR1_PO05_QUAD_F3_R10.hdf5', '4_2_PR2_PO05_QUAD_F3_R10.hdf5', '4_2_PR3_PO05_QUAD_F3_R10.hdf5', '4_2_PR4_PO05_QUAD_F3_R10.hdf5',
                  '4_3_PR1_PO05_QUAD_F3_R10.hdf5', '4_3_PR2_PO05_QUAD_F3_R10.hdf5', '4_3_PR3_PO05_QUAD_F3_R10.hdf5',
                 ]

####  test set for 1234 WITH block
test_Filename_1_x = ['test_1_2_PRx_PO05_QUAD_F3_R10.hdf5', 'test_1_3_PRx_PO05_QUAD_F3_R10.hdf5', 'test_1_4_PRx_PO05_QUAD_F3_R10.hdf5']
test_Filename_2_x = ['test_2_1_PRx_PO05_QUAD_F3_R10.hdf5', 'test_2_3_PRx_PO05_QUAD_F3_R10.hdf5', 'test_2_4_PRx_PO05_QUAD_F3_R10.hdf5']
test_Filename_3_x = ['test_3_1_PRx_PO05_QUAD_F3_R10.hdf5', 'test_3_2_PRx_PO05_QUAD_F3_R10.hdf5', 'test_3_4_PRx_PO05_QUAD_F3_R10.hdf5']
test_Filename_4_x = ['test_4_1_PRx_PO05_QUAD_F3_R10.hdf5', 'test_4_2_PRx_PO05_QUAD_F3_R10.hdf5', 'test_4_3_PRx_PO05_QUAD_F3_R10.hdf5']


############################################################################
#### produce and save trajectory Training set
############################################################################

############   regression mode    ############
'''
dataMat_org, dataMat_x_org, dataMat_y_org = read.read_hdf5_auto_v2(train_path_1234, train_Filename, train_step_size = 3)
##  save training set into file
training_set_xy_file = 'training_set_xy_1234Block' + last_steps + file_format
training_set_x_file = 'training_set_x_1234Block' + last_steps + file_format
training_set_y_file = 'training_set_y_1234Block' + last_steps + file_format
read.save_data(dataMat_org, train_path_1234 + training_set_xy_file)
read.save_data(dataMat_x_org, train_path_1234 + training_set_x_file)
read.save_data(dataMat_y_org, train_path_1234 + training_set_y_file)
'''

############   classification mode   ############
'''
train_posSpeed_set, train_destination_set = read.read_hdf5_auto_v3(train_path_1234, train_Filename, train_step_size = step_size, mode = machine_mode)
training_posSpeed_set_file = 'training_posSpeed_set_1234Block' + last_steps + file_format
train_destination_set_file = 'training_destination_set_1234Block'  + last_steps + file_format
read.save_data(train_posSpeed_set, train_path_1234 + training_posSpeed_set_file)
read.save_data(train_destination_set, train_path_1234 + train_destination_set_file)
'''

############################################################################
#### produce and save trajectory Test set
############################################################################


############   regression mode   ############
'''
dataMat_test_1_x, dataMat_x_test_1_x, dataMat_y_test_1_x = read.read_hdf5_auto_v2(test_path, test_Filename_1_x, train_step_size = 1, mode = machine_mode)
dataMat_test_2_x, dataMat_x_test_2_x, dataMat_y_test_2_x = read.read_hdf5_auto_v2(test_path, test_Filename_2_x, train_step_size = 1, mode = machine_mode)
dataMat_test_3_x, dataMat_x_test_3_x, dataMat_y_test_3_x = read.read_hdf5_auto_v2(test_path, test_Filename_3_x, train_step_size = 1, mode = machine_mode)
dataMat_test_4_x, dataMat_x_test_4_x, dataMat_y_test_4_x = read.read_hdf5_auto_v2(test_path, test_Filename_4_x, train_step_size = 1, mode = machine_mode)

shuffled_x_test_1_x, shuffled_y_test_1_x  = read.shuffle_data(dataMat_x_test_1_x, dataMat_y_test_1_x)
shuffled_x_test_2_x, shuffled_y_test_2_x  = read.shuffle_data(dataMat_x_test_2_x, dataMat_y_test_2_x)
shuffled_x_test_3_x, shuffled_y_test_3_x  = read.shuffle_data(dataMat_x_test_3_x, dataMat_y_test_3_x)
shuffled_x_test_4_x, shuffled_y_test_4_x  = read.shuffle_data(dataMat_x_test_4_x, dataMat_y_test_4_x)

dataMat_x_test = shuffled_x_test_1_x[0:700, :]
dataMat_x_test = np.row_stack((dataMat_x_test, shuffled_x_test_2_x[0:700, :]))
dataMat_x_test = np.row_stack((dataMat_x_test, shuffled_x_test_3_x[0:700, :]))
dataMat_x_test = np.row_stack((dataMat_x_test, shuffled_x_test_4_x[0:700, :]))

dataMat_y_test = shuffled_y_test_1_x[0:700, :]
dataMat_y_test = np.row_stack((dataMat_y_test, shuffled_y_test_2_x[0:700, :]))
dataMat_y_test = np.row_stack((dataMat_y_test, shuffled_y_test_3_x[0:700, :]))
dataMat_y_test = np.row_stack((dataMat_y_test, shuffled_y_test_4_x[0:700, :]))

'''

##  save regression test set into file
'''
test_set_x_file = 'test_set_x_1234Block'  + last_steps + file_format
test_set_y_file = 'test_set_y_1234Block'  + last_steps + file_format
read.save_data(dataMat_x_test, train_path_1234 + test_set_x_file)
read.save_data(dataMat_y_test, train_path_1234 + test_set_y_file)
'''

## confirm the dataMat_org by visualization
#plot.plot_1234(dataMat_org[::2,:], origShow=True, predShow =False, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)






############   classification mode    ############
'''
test_1_posSpeed_set, test_1_destination_set = read.read_hdf5_auto_v3(test_path_1234, test_Filename_1_x, train_step_size = step_size, mode = machine_mode)
test_2_posSpeed_set, test_2_destination_set = read.read_hdf5_auto_v3(test_path_1234, test_Filename_2_x, train_step_size = step_size, mode = machine_mode)
test_3_posSpeed_set, test_3_destination_set = read.read_hdf5_auto_v3(test_path_1234, test_Filename_3_x, train_step_size = step_size, mode = machine_mode)
test_4_posSpeed_set, test_4_destination_set = read.read_hdf5_auto_v3(test_path_1234, test_Filename_4_x, train_step_size = step_size, mode = machine_mode)

test_1_posSpeed_set_shuffled, test_1_destination_set_shuffled = read.shuffle_data(test_1_posSpeed_set, test_1_destination_set)
test_2_posSpeed_set_shuffled, test_2_destination_set_shuffled = read.shuffle_data(test_2_posSpeed_set, test_2_destination_set)
test_3_posSpeed_set_shuffled, test_3_destination_set_shuffled = read.shuffle_data(test_3_posSpeed_set, test_3_destination_set)
test_4_posSpeed_set_shuffled, test_4_destination_set_shuffled = read.shuffle_data(test_4_posSpeed_set, test_4_destination_set)

test_posSpeed_set = test_1_posSpeed_set_shuffled[0:700, :]
test_posSpeed_set = np.row_stack((test_posSpeed_set, test_2_posSpeed_set_shuffled[0:700, :]))
test_posSpeed_set = np.row_stack((test_posSpeed_set, test_3_posSpeed_set_shuffled[0:700, :]))
test_posSpeed_set = np.row_stack((test_posSpeed_set, test_4_posSpeed_set_shuffled[0:700, :]))

test_destination_set = test_1_destination_set_shuffled[0:700, :]
test_destination_set = np.row_stack((test_destination_set, test_2_destination_set_shuffled[0:700, :]))
test_destination_set = np.row_stack((test_destination_set, test_3_destination_set_shuffled[0:700, :]))
test_destination_set = np.row_stack((test_destination_set, test_4_destination_set_shuffled[0:700, :]))
'''

##  save classification test set into file
'''
test_posSpeed_set_file = 'test_1234Block_posSpeed_set' + last_steps + file_format
test_destination_set_file = 'test_1234Block_destination_set' + last_steps + file_format
read.save_data(test_posSpeed_set, test_path_1234 + test_posSpeed_set_file)
read.save_data(test_destination_set, test_path_1234 + test_destination_set_file)
'''












#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###  data bikeSpeedVsIq_train
'''
dataList_train_bike = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_train.txt"); dataMat_train_bike = np.mat(dataList_train_bike)
dataList_test_bike = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_test.txt"); dataMat_test_bike = np.mat(dataList_test_bike)
'''
###  data 4: show the highlight of Model Tree
'''
#dataList_data4 = reg.loadDataList("regression/lab/input/data4.txt"); dataMat_data4 = np.mat(dataList_data4)
#reg.showStdLinReg(dataList_data4 )
'''
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


############################################################################
###################  load training and test set   #########################
############################################################################
### street 1234

training_set_xy_file = 'training_set_xy_1234Block' + last_steps + file_format
training_set_x_filename = "training_set_x_1234Block" + last_steps + file_format
training_set_y_filename = "training_set_y_1234Block" + last_steps + file_format
test_set_x_filename = "test_set_x_1234Block" + last_steps + file_format
test_set_y_filename = "test_set_y_1234Block" + last_steps + file_format  

dataMat_org = read.load_data(train_path_1234 + training_set_xy_file)
dataMat_x_org = read.load_data(train_path_1234 + training_set_x_filename)
dataMat_y_org = read.load_data(train_path_1234 + training_set_y_filename)
dataMat_x_test = read.load_data(test_path_1234 + test_set_x_filename)
dataMat_y_test = read.load_data(test_path_1234 + test_set_y_filename)
dataMat_test = np.mat(np.zeros((dataMat_x_test.shape[0],dataMat_x_test.shape[1]+1)))
dataMat_test[:,:-1] = dataMat_x_test
dataMat_test[:,-1] = dataMat_y_test[:,-1]



#plot.plot_1234(dataMat_org[::2,:], origShow=True, predShow =False, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)

############################################################################
####################    Standard Linear Regression    ######################
############################################################################
'''
ws,X,Y = reg.stdLinReg(dataMat_train)
yMat_pred = reg.pred_stdLinReg(dataMat_test,ws)
reg.showStdLinReg(dataList_test, yMat_pred)

weight_x,X1,Y1 = reg.stdLinReg(dataMat_x_org)
yMat_pred_x = reg.pred_stdLinReg(dataMat_x_org, weight_x)

weight_y,X2,Y2 = reg.stdLinReg(dataMat_y_org)
yMat_pred_y = reg.pred_stdLinReg(dataMat_y_org, weight_y)

#### evaluate the Linear regression
print("correlation coefficients (dataMat_x_org, LinReg): \n\n", np.corrcoef(yMat_pred_x.T, dataMat_x_org[:,-1].T))
print('\nSquared error (dataMat_x_org, LinReg): ', reg.calError(dataMat_x_org[:,-1].flatten().A[0].tolist(), yMat_pred_x.T.A))

print("correlation coefficients (dataMat_y_org, LinReg): \n\n", np.corrcoef(yMat_pred_y.T, dataMat_y_org[:,-1].T))
print('\nSquared error (dataMat_y_org, LinReg): ', reg.calError(dataMat_y_org[:,-1].flatten().A[0].tolist(), yMat_pred_y.T.A))
'''






############################################################################
####################    Locally weighted linear regression    ##############
############################################################################
'''
yArr_pred = reg.lwlr_Test(dataMat_test, dataMat_train_bike, k=1.0)
reg.showLwlr(dataList_test, yArr_pred, k=1.0)

yArr_pred_x = reg.lwlr_Test(dataMat_x_org, dataMat_x_org, k=1)
yArr_pred_y = reg.lwlr_Test(dataMat_y_org, dataMat_y_org, k=1)
print('for k=1, the Error (Test):',reg.calError(dataMat_x_org[:,-1].flatten().A[0].tolist(), yArr_pred_x.T))
print('for k=1, the Error (Test):',reg.calError(dataMat_y_org[:,-1].flatten().A[0].tolist(), yArr_pred_y.T))
'''




############################################################################
####################    Regressino Tree       ##############################
############################################################################

#### build the Regression Tree for trajectory
'''
#### build the Regressino Tree for accX and accY
regTreeX_trained = reg.createTree(dataMat_x_org, stopCond=(0,15))
regTreeY_trained = reg.createTree(dataMat_y_org, stopCond=(0,15))

#### save regression tree into file
regTreeX_trained_file = 'regTreeX_1234Block_trained_0_15' + last_steps + file_format
regTreeY_trained_file = 'regTreeY_1234Block_trained_0_15' + last_steps + file_format
reg.saveTree(regTreeX_trained, model_path + regTreeX_trained_file)
reg.saveTree(regTreeY_trained, model_path + regTreeY_trained_file)
'''

#### build the Regression Tree for bike
'''
modTree_bike_trained = reg.createTree(dataMat_train_bike, reg.modelLeaf, reg.modelErr, stopCond=(0, 20))
#### save Model Tree into file
regTree_bike_file = 'modTreeX_bike_test_pinv' + file_format
reg.saveTree(modTree_bike_trained, regTree_bike_file)
'''


#### build the Regression Tree for data4
'''
regTree_data4_trained = reg.createTree(dataMat_data4, stopCond=(0, 20))
#### save Model Tree into file
regTree_trained_file = 'regTree_data4_0_20' + file_format
reg.saveTree(regTree_data4_trained, regTree_trained_file)
'''




############################################################################
####################    Model Tree       ###################################
############################################################################

##########    build the Model Tree for trajectory   ########
'''
#### build the Model Tree for accX and accY
modTreeX_trained = reg.createTree(dataMat_x_org, reg.modelLeaf, reg.modelErr, stopCond=(0,20))
modTreeY_trained = reg.createTree(dataMat_y_org, reg.modelLeaf, reg.modelErr, stopCond=(0,20))

#### save regression tree into file
modTreeX_trained_file = 'modTreeX_1234Block_trained_0_20' + last_steps + file_format
modTreeY_trained_file = 'modTreeY_1234Block_trained_0_20' + last_steps + file_format
reg.saveTree(modTreeX_trained, model_path + modTreeX_trained_file)
reg.saveTree(modTreeY_trained, model_path + modTreeY_trained_file)
'''

##########   build the Model Tree for data4    ########
'''
modTree_data4_trained = reg.createTree(dataMat_data4, reg.modelLeaf, reg.modelErr, stopCond=(0, 20))
#### save Model Tree into file
modTree_file = 'modTree_data4_0_20' + file_format
reg.saveTree(modTree_data4_trained, modTree_file)
'''

##########   build the Model Tree for bike    ########
'''
modTree_bike_trained = reg.createTree(dataMat_train_bike, reg.modelLeaf, reg.modelErr, stopCond=(0, 20))
#### save Model Tree into file
modTree_file = 'modTree_bike_0_20' + file_format
reg.saveTree(modTree_bike_trained, modTree_file)
'''







############################################################################
########    Gradient descend Tree       #############
############################################################################

##########   build the descend Tree for data4    ########
'''
desTree_trained = reg.createTree(dataMat_data4, reg.gradDesLeaf, reg.gradDesErr,
                              stopCond=(0,20), minPercent = 0.01, epoch = 30) 
#### save descend Tree into file
desTree_file = 'desTree_data4_0_20_min001_epoch30' + file_format
reg.saveTree(desTree_trained, desTree_file)
'''

##########   build the descend Tree for bike    ########
'''
desTree_trained_bike = reg.createTree(dataMat_train_bike, reg.gradDesLeaf, reg.gradDesErr,
                              stopCond=(0,20), minPercent = 0.01, epoch = 1000) 
#### save descend Tree into file
desTree_file = 'desTree_bike_0_20_min001_epoch30' + file_format
reg.saveTree(desTree_trained_bike, desTree_file)
'''


##########   build the descend Tree for trajectory    ########
'''
#### build the Model Tree for accX and accY
desTreeX_trained = reg.createTree(dataMat_x_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,20), minPercent = 10, epoch = 10) 
desTreeY_trained = reg.createTree(dataMat_y_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,20), minPercent = 10, epoch = 10) 

#### save descend tree into file
desTreeX_trained_file = 'desTreeX_trained_0_25_min10_epoch10' + file_format
desTreeY_trained_file = 'desTreeY_trained_0_25_min10_epoch10' + file_format
reg.saveTree(desTreeX_trained, desTreeX_trained_file)
reg.saveTree(desTreeY_trained, desTreeY_trained_file)
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
layers_dims = [len(train_x),20, 4]  # version: 1

#### build  model
class_model_trained = NN.L_layer_model(train_x, train_y_destination, layers_dims, learning_mode = "classification", optimizer = "adam", 
                                          enable_mini_batch = True,  learning_rate = 0.01, num_iterations = 500, print_cost = True)  

#### save neural network into file
regNN_1234_model_file = 'regNN_1234Block_classModel_trained_v3' + last_steps + file_format
read.save_data(class_model_trained, model_path_1234 + regNN_1234_model_file)
'''

#########################################
###  train bike with neural network   ###
#########################################
'''
norm_parameters, train_x = regNN.normData(dataMat_train_bike[:,:-1])
train_y = dataMat_train_bike[:,-1].T


#### hidden layers
layers_dims = [len(train_x), 30,20,5, 1]  
# layers_dims = [len(train_x), 40, 40, 40, 40, 40, 40, 1] #  

#### build model
parameters = regNN.L_layer_model(train_x, train_y, layers_dims, optimizer = "adam", enable_mini_batch = False, 
                                 learning_rate = 0.01, num_iterations = 1000, print_cost = True, plot_cost = True) 


#### save regression network into file
regression_network_file = 'regNN_bike_testOpt_with_minibatch' + file_format
regNN.saveNetwork(parameters, model_path + regression_network_file)
'''


#########################################
###  train data4 with neural network  ###
#########################################
'''
norm_parameters, train_x = regNN.normData(dataMat_data4[:,:-1])
train_y = dataMat_data4[:,-1].T
print ("train_x's shape: " + str(train_x.shape))
print ("train_y's shape: " + str(train_y.shape))
#layers_dims = [len(train_x), 20, 30, 10, 1]  

layers_dims = [len(train_x), 5, 5, 1] #  
parameters = regNN.L_layer_model(train_x, train_y, layers_dims, learning_rate = 10, num_iterations = 1000, 
                                 print_cost = True, regMode = True, foreAct = 'relu', backAct = 'relu') #'sigmoid'
#### save regression tree into file
regression_network_file = 'regNN_data4_3' + file_format
regNN.saveNetwork(parameters, model_path + regression_network_file)
'''



