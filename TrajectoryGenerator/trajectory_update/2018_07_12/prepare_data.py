import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import regNetwork as regNN
import dnn_app_utils_v2 as dnn
###############################   read data   ######################################
### average walk speed is 1.2 m/s.  so the distance will be 6 meters in 5 seconds. 
### the scale of simulation is: 6 meters is equal to 100 pixels, which in turn gives the speed with equation:  speed(m/s) = pixels in speed / 16.6

#path = 'samples/'
#Filename = ['mid-1_uni_PR10_PO10_F5_R10.hdf5','mid0_uni_PR10_PO10_F5_R10.hdf5','mid+1_uni_PR10_PO10_F5_R10.hdf5', 'mid+05_uni_PR10_PO10_F5_R10.hdf5', 'mid-05_uni_PR10_PO10_F5_R10.hdf5']

train_path = 'samples/1234/'
train_Filename = ['1_2_PR1_PO0_LIN_F4_R10.hdf5', '1_2_PR2_PO0_LIN_F4_R10.hdf5', '1_2_PR3_PO0_LIN_F4_R10.hdf5', '1_2_PR4_PO0_LIN_F4_R10.hdf5',
            '1_3_PR1_PO0_LIN_F4_R10.hdf5', '1_3_PR2_PO0_LIN_F4_R10.hdf5', '1_3_PR3_PO0_LIN_F4_R10.hdf5', '1_3_PR4_PO0_LIN_F4_R10.hdf5', '1_3_PR5_PO0_LIN_F4_R10.hdf5', '1_3_PR6_PO0_LIN_F4_R10.hdf5',
            '1_4_PR1_PO0_LIN_F4_R10.hdf5', '1_4_PR2_PO0_LIN_F4_R10.hdf5', '1_4_PR3_PO0_LIN_F4_R10.hdf5', '1_4_PR4_PO0_LIN_F4_R10.hdf5', '1_4_PR5_PO0_LIN_F4_R10.hdf5', '1_4_PR6_PO0_LIN_F4_R10.hdf5',
            '2_1_PR1_PO0_LIN_F4_R10.hdf5', '2_1_PR2_PO0_LIN_F4_R10.hdf5', '2_1_PR3_PO0_LIN_F4_R10.hdf5', '2_1_PR4_PO0_LIN_F4_R10.hdf5',
            '2_3_PR1_PO0_LIN_F4_R10.hdf5', '2_3_PR2_PO0_LIN_F4_R10.hdf5', '2_3_PR3_PO0_LIN_F4_R10.hdf5', '2_3_PR4_PO0_LIN_F4_R10.hdf5', '2_3_PR5_PO0_LIN_F4_R10.hdf5', '2_3_PR6_PO0_LIN_F4_R10.hdf5',
            '2_4_PR1_PO0_LIN_F4_R10.hdf5', '2_4_PR2_PO0_LIN_F4_R10.hdf5', '2_4_PR3_PO0_LIN_F4_R10.hdf5', '2_4_PR4_PO0_LIN_F4_R10.hdf5', '2_4_PR5_PO0_LIN_F4_R10.hdf5', '2_4_PR6_PO0_LIN_F4_R10.hdf5',
            '3_1_PR1_PO0_LIN_F4_R10.hdf5', '3_1_PR2_PO0_LIN_F4_R10.hdf5', '3_1_PR3_PO0_LIN_F4_R10.hdf5', '3_1_PR4_PO0_LIN_F4_R10.hdf5', '3_1_PR5_PO0_LIN_F4_R10.hdf5', '3_1_PR6_PO0_LIN_F4_R10.hdf5',
            '3_2_PR1_PO0_LIN_F4_R10.hdf5', '3_2_PR2_PO0_LIN_F4_R10.hdf5', '3_2_PR3_PO0_LIN_F4_R10.hdf5', '3_2_PR4_PO0_LIN_F4_R10.hdf5', '3_2_PR5_PO0_LIN_F4_R10.hdf5', '3_2_PR6_PO0_LIN_F4_R10.hdf5',
            '3_4_PR1_PO0_LIN_F4_R10.hdf5', '3_4_PR2_PO0_LIN_F4_R10.hdf5', '3_4_PR3_PO0_LIN_F4_R10.hdf5', '3_4_PR4_PO0_LIN_F4_R10.hdf5',
            '4_1_PR1_PO0_LIN_F4_R10.hdf5', '4_1_PR2_PO0_LIN_F4_R10.hdf5', '4_1_PR3_PO0_LIN_F4_R10.hdf5', '4_1_PR4_PO0_LIN_F4_R10.hdf5', '4_1_PR5_PO0_LIN_F4_R10.hdf5', '4_1_PR6_PO0_LIN_F4_R10.hdf5',
            '4_2_PR1_PO0_LIN_F4_R10.hdf5', '4_2_PR2_PO0_LIN_F4_R10.hdf5', '4_2_PR3_PO0_LIN_F4_R10.hdf5', '4_2_PR4_PO0_LIN_F4_R10.hdf5', '4_2_PR5_PO0_LIN_F4_R10.hdf5', '4_2_PR6_PO0_LIN_F4_R10.hdf5',
            '4_3_PR1_PO0_LIN_F4_R10.hdf5', '4_3_PR2_PO0_LIN_F4_R10.hdf5', '4_3_PR3_PO0_LIN_F4_R10.hdf5', '4_3_PR4_PO0_LIN_F4_R10.hdf5',
            ]
dataMat_org, dataMat_x_org, dataMat_y_org = read.read_hdf5_auto(train_path, train_Filename)

## confirm the dataMat_org by visualization
#plot.plotTrack_mult(dataMat_org)



###  data bikeSpeedVsIq_train
'''
dataList_train_bike = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_train.txt"); dataMat_train_bike = np.mat(dataList_train_bike)
dataList_test_bike = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_test.txt"); dataMat_test_bike = np.mat(dataList_test_bike)
'''


###  data 4: show the highlight of Model Tree
#dataList_data4 = reg.loadDataList("regression/lab/input/data4.txt"); dataMat_data4 = np.mat(dataList_data4)

#reg.showStdLinReg(dataList_data4 )




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
regTreeX_trained_file = 'regTreeX_trained_0_15.txt'
regTreeY_trained_file = 'regTreeY_trained_0_15.txt'
reg.saveTree(regTreeX_trained, regTreeX_trained_file)
reg.saveTree(regTreeY_trained, regTreeY_trained_file)
'''

#### build the Regression Tree for bike
'''
modTree_bike_trained = reg.createTree(dataMat_train_bike, reg.modelLeaf, reg.modelErr, stopCond=(0, 20))
#### save Model Tree into file
regTree_bike_file = 'modTreeX_bike_0_20.dat'
reg.saveTree(modTree_bike_trained, regTree_bike_file)
'''


#### build the Regression Tree for data4
'''
regTree_data4_trained = reg.createTree(dataMat_data4, stopCond=(0, 20))
#### save Model Tree into file
regTree_trained_file = 'regTree_data4_0_20.dat'
reg.saveTree(regTree_data4_trained, regTree_trained_file)
'''




############################################################################
####################    Model Tree       ###################################
############################################################################

##########    build the Model Tree for trajectory   ########
'''
#### build the Model Tree for accX and accY
modTreeX_trained = reg.createTree(dataMat_x_org, reg.modelLeaf, reg.modelErr, stopCond=(0,15))
modTreeY_trained = reg.createTree(dataMat_y_org, reg.modelLeaf, reg.modelErr, stopCond=(0,15))

#### save regression tree into file
modTreeX_trained_file = 'modTreeX_trained_0_15.dat'
modTreeY_trained_file = 'modTreeY_trained_0_15.dat'
reg.saveTree(modTreeX_trained, modTreeX_trained_file)
reg.saveTree(modTreeY_trained, modTreeY_trained_file)

'''


##########   build the Model Tree for data4    ########
'''
modTree_data4_trained = reg.createTree(dataMat_data4, reg.modelLeaf, reg.modelErr, stopCond=(0, 20))
#### save Model Tree into file
modTree_file = 'modTree_data4_0_20.dat'
reg.saveTree(modTree_data4_trained, modTree_file)
'''

##########   build the Model Tree for bike    ########
'''
modTree_bike_trained = reg.createTree(dataMat_train_bike, reg.modelLeaf, reg.modelErr, stopCond=(0, 20))
#### save Model Tree into file
modTree_file = 'modTree_bike_0_20.dat'
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
desTree_file = 'desTree_data4_0_20_min001_epoch30.dat'
reg.saveTree(desTree_trained, desTree_file)
'''

##########   build the descend Tree for bike    ########
'''
desTree_trained_bike = reg.createTree(dataMat_train_bike, reg.gradDesLeaf, reg.gradDesErr,
                              stopCond=(0,20), minPercent = 0.01, epoch = 1000) 
#### save descend Tree into file
desTree_file = 'desTree_bike_0_20_min001_epoch30.dat'
reg.saveTree(desTree_trained_bike, desTree_file)
'''


##########   build the descend Tree for trajectory    ########

#### build the Model Tree for accX and accY
desTreeX_trained = reg.createTree(dataMat_x_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,20), minPercent = 10, epoch = 10) 
desTreeY_trained = reg.createTree(dataMat_y_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,20), minPercent = 10, epoch = 10) 

#### save descend tree into file
desTreeX_trained_file = 'desTreeX_trained_0_25_min10_epoch10.dat'
desTreeY_trained_file = 'desTreeY_trained_0_25_min10_epoch10.dat'
reg.saveTree(desTreeX_trained, desTreeX_trained_file)
reg.saveTree(desTreeY_trained, desTreeY_trained_file)









############################################################################
####################    neural network       ###############################
############################################################################
'''
norm_parameters_x, train_x = regNN.normData(dataMat_x_org[:,:-1])
norm_parameters_y, train_y = regNN.normData(dataMat_y_org[:,:-1])
train_x_target = dataMat_x_org[:,-1].T
train_y_target = dataMat_y_org[:,-1].T


#layers_dims = [len(train_x), 20, 30, 40, 40, 40, 20, 10, 1]  # version: 0
layers_dims = [len(train_x), 20, 30, 10, 1]  # version: 1
parameters_x = regNN.L_layer_model(train_x, train_x_target, layers_dims, learning_rate = 1000, num_iterations = 4000, 
                                 print_cost = True, plot_cost = False, regMode = True, foreAct = 'relu', backAct = 'relu')  


parameters_y = regNN.L_layer_model(train_y, train_y_target, layers_dims, learning_rate = 1000, num_iterations = 4000, 
                                 print_cost = True, plot_cost = False, regMode = True, foreAct = 'relu', backAct = 'relu')  



#### save neural network into file
regNN_1234_x_file = 'regNN_1234_x_trained_4.txt'
regNN_1234_y_file = 'regNN_1234_y_trained_4.txt'
regNN.saveNetwork(parameters_x, regNN_1234_x_file)
regNN.saveNetwork(parameters_y, regNN_1234_y_file)



'''

#########################################
###  train bike with neural network   ###
#########################################
'''
norm_parameters, train_x = regNN.normData(dataMat_train[:,:-1])
train_y = dataMat_train[:,-1].T
print ("train_x's shape: " + str(train_x.shape))
print ("train_y's shape: " + str(train_y.shape))
#layers_dims = [len(train_x), 20, 30, 10, 1]  

layers_dims = [len(train_x), 40, 40, 40, 40, 40, 40, 1] #  
parameters = regNN.L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.1, num_iterations = 1000, 
                                 print_cost = True, regMode = True, foreAct = 'relu', backAct = 'relu') #'sigmoid'
#### save regression tree into file
regression_network_file = 'regNN_bike_overfitting_2.dat'
regNN.saveNetwork(parameters, regression_network_file)
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
regression_network_file = 'regNN_data4_3.dat'
regNN.saveNetwork(parameters, regression_network_file)
'''




################################################
#######   Content from Coursera-Notebook  ######
################################################
'''
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
layers_dims = [12288,  5, 1] #  5-layer model
parameters = dnn.L_layer_model(train_x, train_y, layers_dims, num_iterations = 10, print_cost = True)
#parameters = regNN.L_layer_model(train_x, train_y, layers_dims, num_iterations = 10, print_cost = True, regMode = False)
'''

