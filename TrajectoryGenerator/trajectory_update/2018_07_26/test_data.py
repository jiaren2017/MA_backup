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
train_path = 'samples/1234/training_test_set/1234_with_block/';  
##  training set for 1234 WITH block
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
##  test set for 1234 WITH block
test_path = 'samples/1234/training_test_set/1234_with_block/';  
test_Filename = ['test_1_PR2_PO05_QUAD_F3_R6.hdf5', 'test_2_PR2_PO05_QUAD_F3_R6.hdf5', 'test_3_PR2_PO05_QUAD_F3_R6.hdf5', 'test_4_PR2_PO05_QUAD_F3_R6.hdf5'
                ]



'''
##  training set for 1234 WITHOUT block
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

##  test set for 1234 WITHOUT block
test_path = 'samples/1234/'; 
test_Filename = ['test_1_x_PR2_PO0_LIN_F4_R5.hdf5', 'test_2_x_PR2_PO0_LIN_F4_R5.hdf5', 'test_3_x_PR2_PO0_LIN_F4_R5.hdf5', 'test_4_x_PR2_PO0_LIN_F4_R5.hdf5',
                 'test_1_x_PR2_PO0_LIN_F4_R2.hdf5', 'test_2_x_PR2_PO0_LIN_F4_R2.hdf5', 'test_3_x_PR2_PO0_LIN_F4_R2.hdf5', 'test_4_x_PR2_PO0_LIN_F4_R2.hdf5',
                 ]

'''



model_path = "trained_model/2018_07_26/"



dataMat_org, dataMat_x_org, dataMat_y_org = read.read_hdf5_auto_v2(train_path, train_Filename, train_step_size = 1)
dataMat_test, dataMat_x_test, dataMat_y_test = read.read_hdf5_auto_v2(test_path, test_Filename, train_step_size = 1)
#plot.plotTrack_mult(dataMat_test)  # check the data





###  data bikeSpeedVsIq_train
'''
dataList_train_bike = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_train.txt"); dataMat_train_bike = np.mat(dataList_train_bike)
dataList_test_bike = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_test.txt"); dataMat_test_bike = np.mat(dataList_test_bike)
'''

###  data 4
'''
dataList_data4 = reg.loadDataList("regression/lab/input/data4.txt"); dataMat_data4 = np.mat(dataList_data4)
'''





############################################################################
###################  load Tree data   #########################
############################################################################

############## Regression Tree  #######

#### trajectory
'''
regTreeX_trained_file = 'regTreeX_trained_0_15.txt';    regTreeX_trained = reg.loadTree(regTreeX_trained_file)
regTreeY_trained_file = 'regTreeY_trained_0_15.txt';    regTreeY_trained = reg.loadTree(regTreeY_trained_file)
'''

#### data 4
'''
regTree_trained_file = 'regTree_data4_0_20.dat';    regTree_data4_trained = reg.loadTree(regTree_trained_file)
reg.createPlot(regTree_data4_trained)
'''

############## Model Tree  #######

#### data 4
#modTree_file = 'modTree_data4_0_20.dat';    modTree_data4_trained = reg.loadTree(modTree_file)

#### bike
#modTree_file = 'modTreeX_bike_test_pinv.dat';    modTree_bike_trained = reg.loadTree(modTree_file)



############## Descent Tree  #######
#### data 
#desTree_file = 'desTree_data4_0_20_min001_epoch30.dat';    desTree_data4_trained = reg.loadTree(desTree_file)

#### bike
#desTree_file = 'desTree_bike_0_20_min001_epoch30.dat';    desTree_bike_trained = reg.loadTree(desTree_file)





############################################################################
##################### load neural network data   ################## 
############################################################################

#####  load data for trajectory  #####
'''
regNN_1234_x_file = 'regNN_1234_x_trained_v5_adam_minibatchON_last3.dat'; regNN_parameters_x = regNN.loadNNetwork(model_path + regNN_1234_x_file)
regNN_1234_y_file = 'regNN_1234_y_trained_v5_adam_minibatchON_last3.dat'; regNN_parameters_y = regNN.loadNNetwork(model_path + regNN_1234_y_file)
norm_parameters_x, train_x = regNN.normData(dataMat_x_org[:,:-1]); train_x_target = dataMat_x_org[:,-1].T
norm_parameters_y, train_y = regNN.normData(dataMat_y_org[:,:-1]); train_y_target = dataMat_y_org[:,-1].T
'''

#####  load data for bike  #####
'''
regression_network_file =  'regNN_bike_testOpt_with_minibatch.dat'
parameters_bike = regNN.loadNNetwork(model_path + regression_network_file)
norm_parameters_bike, train_x_bike = regNN.normData(dataMat_train_bike[:,:-1])
'''


#####  load data for data4  #####
'''
regression_network_file =  'regNN_data4_3.dat'
parameters_data4 = regNN.loadNNetwork(regression_network_file)
norm_parameters_data4, train_x_data4 = regNN.normData(dataMat_data4[:,:-1])
'''










############################################################################
####################    evaluate Regressino Tree       #####################
############################################################################

#### evaluate trajectory
'''
## evaluate with train_x ----------------------------------
print('\n\n---------------------   evaluate regression tree   ------------------------------------')
train_accArr_x_regTree = reg.createForeCast(regTreeX_trained, dataMat_x_org[:,:-1])    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (data: train, target: accX, mode: regTree): ", np.corrcoef(train_accArr_x_regTree, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (data: train, target: accX, mode: regTree): ", reg.calError(dataMat_x_org[:,-1].flatten().A[0], train_accArr_x_regTree))

train_accArr_Y_regTree = reg.createForeCast(regTreeY_trained, dataMat_y_org[:,:-1])  
print("\ncorrelation coefficients (data: train, target: accY, mode: regTree): ", np.corrcoef(train_accArr_Y_regTree, dataMat_y_org[:,-1], rowvar=0)[0,1])
print("\nSquared error (data: train, target: accY, mode: regTree): ", reg.calError(dataMat_y_org[:,-1].flatten().A[0], train_accArr_Y_regTree))


## evaluate with test_x ----------------------------------
test_accArr_x_regTree = reg.createForeCast(regTreeX_trained, dataMat_x_test[:,:-1])    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (data: test, target: accX, mode: regTree): ", np.corrcoef(test_accArr_x_regTree, dataMat_x_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (data: test, target: accX, mode: regTree): ", reg.calError(dataMat_x_test[:,-1].flatten().A[0], test_accArr_x_regTree))

test_accArr_Y_regTree = reg.createForeCast(regTreeY_trained, dataMat_y_test[:,:-1])  
print("\ncorrelation coefficients (data: test, target: accY, mode: regTree): ", np.corrcoef(test_accArr_Y_regTree, dataMat_y_test[:,-1], rowvar=0)[0,1])
print("\nSquared error (data: test, target: accY, mode: regTree): ", reg.calError(dataMat_y_test[:,-1].flatten().A[0], test_accArr_Y_regTree))
'''

#### evaulate data4
'''
yArr_pred_test = reg.createForeCast(regTree_data4_trained, dataMat_data4[:,0])
reg.showTree(dataList_data4, yArr_pred_test, mode = "regTree")
'''


############################################################################
####################    evaluate Model Tree       ##########################
############################################################################
#### evaulate bike data
'''
yArr_pred_test = reg.createForeCast(modTree_bike_trained, dataMat_test_bike[:,0], reg.modelTreeEval)
reg.showTree(dataList_test_bike, yArr_pred_test, mode = "modTree")
'''

#### evaulate data4
'''
yArr_pred_test = reg.createForeCast(modTree_data4_trained, dataMat_data4[:,0], reg.modelTreeEval)
reg.showTree(dataList_data4, yArr_pred_test, mode = "modTree")
'''





############################################################################
########    evaluate Gradient descend Tree       #############
############################################################################

#### evaulate data4
'''
yArr_pred_test = reg.createForeCast(desTree_data4_trained, dataMat_data4[:,0], reg.gradDesTreeEval)
reg.showTree(dataList_data4, yArr_pred_test, mode = "gradDesTree")
'''


#### evaulate bike data
'''
yArr_pred_test = reg.createForeCast(desTree_bike_trained, dataMat_test_bike[:,0], reg.gradDesTreeEval)
reg.showTree(dataList_test_bike, yArr_pred_test, mode = "gradDesTree")
'''


##########   build the descend Tree for trajectory    ########
'''
#### build the Model Tree for accX and accY
modTreeX_trained = reg.createTree(dataMat_x_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,10), minPercent = 0.001, epoch = 50) 
modTreeY_trained = reg.createTree(dataMat_y_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,10), minPercent = 0.001, epoch = 50) 

#### evaluate the Model Tree
accArr_x = reg.createForeCast(modTreeX_trained, dataMat_x_org[:,:-1], reg.gradDesTreeEval)    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (accX, regTree): ", np.corrcoef(accArr_x, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (accX, regTree): ", reg.calError(dataMat_x_org[:,-1].flatten().A[0], accArr_x))

accArr_Y = reg.createForeCast(modTreeY_trained, dataMat_y_org[:,:-1], reg.gradDesTreeEval)  
print("\ncorrelation coefficients (accY, regTree): ", np.corrcoef(accArr_Y, dataMat_y_org[:,-1], rowvar=0)[0,1])
print("\nSquared error (accY, regTree): ", reg.calError(dataMat_y_org[:,-1].flatten().A[0], accArr_Y))

'''



############################################################################
####################    evaluate nerual network       ######################
############################################################################

####  evaulate trajectory   

'''
## evaluate with train_x ----------------------------------
print('\n\n---------------------   evaluate neural network   ------------------------------------')
train_accArr_x_regNN = regNN.network_predict(train_x, regNN_parameters_x, regMode = True, foreAct = 'relu')
train_accArr_x_regNN = np.array(train_accArr_x_regNN).ravel()  # convert matrix to array !!
print("\ncorrelation coefficients (data: train, target: accX, mode: regNN): ", np.corrcoef(train_accArr_x_regNN, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (data: train, target: accX, mode: regNN): ", reg.calError(dataMat_x_org[:,-1].flatten().A[0], train_accArr_x_regNN))

train_accArr_y_regNN = regNN.network_predict(train_y, regNN_parameters_y, regMode = True, foreAct = 'relu')
train_accArr_y_regNN = np.array(train_accArr_y_regNN).ravel()
print("\ncorrelation coefficients (data: train, target: accY, mode: regNN): ", np.corrcoef(train_accArr_y_regNN, dataMat_y_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (data: train, target: accY, mode: regNN): ", reg.calError(dataMat_y_org[:,-1].flatten().A[0], train_accArr_y_regNN))


## evaluate with test_x ----------------------------------  normalization !!
dataMat_x_test_regNN = dataMat_x_test[:,:-1].T 
dataMat_y_test_regNN = dataMat_y_test[:,:-1].T 

test_accArr_x_regNN = regNN.network_predict(dataMat_x_test_regNN, regNN_parameters_x, norm_parameters_NN = norm_parameters_x,  regMode = True, foreAct = 'relu')
test_accArr_x_regNN = np.array(test_accArr_x_regNN).ravel()  # convert matrix to array !!
print("\ncorrelation coefficients (data: test, target: accX, mode: regNN): ", np.corrcoef(test_accArr_x_regNN, dataMat_x_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (data: test, target: accX, mode: regNN): ", reg.calError(dataMat_x_test[:,-1].flatten().A[0], test_accArr_x_regNN))

test_accArr_y_regNN = regNN.network_predict(dataMat_y_test_regNN, regNN_parameters_y, norm_parameters_NN = norm_parameters_y, regMode = True, foreAct = 'relu')
test_accArr_y_regNN = np.array(test_accArr_y_regNN).ravel()
print("\ncorrelation coefficients (data: test, target: accY, mode: regNN): ", np.corrcoef(test_accArr_y_regNN, dataMat_y_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (data: test, target: accY, mode: regNN): ", reg.calError(dataMat_y_test[:,-1].flatten().A[0], test_accArr_y_regNN))

'''


############################################################################
#########################       Test       #################################
############################################################################


#### use original data
#random_pos = random.randint(0, dataMat_org.shape[0]-1)
#dataMat_test = dataMat_org[random_pos,0:-2].T           # make the starting point as test data
#dataMat_test[:,2:] = 0                     # define the speed


#### customize the test data
#dataMat_test = np.mat(np.array([380,250,-100,-100]))


############################################################################
############################################################################
#### test the Regression Tree !!
#dataMat_regPred = reg.createForeCast(regTreeX_trained, dataMat_test, modelEval = reg.regTreeEval, stepPred=True, 
#                                       treeX_trained=regTreeX_trained, treeY_trained=regTreeY_trained, numSteps=10)

############################################################################
############################################################################
#### test the Model Tree !!
#dataMat_ModPred = reg.createForeCast(modTreeX_trained, dataMat_test, modelEval = reg.gradDesTreeEval, stepPred=True,treeX_trained=modTreeX_trained, treeY_trained=modTreeY_trained, numSteps=10) 


############################################################################
############################################################################
#### test the neural network !!      
####  test walk trace 
'''
## for the neural network, features are rows vectors and samples are column vectors
#dataMat_test = dataMat_test.T 
## predict!
dataMat_regNN_Pred = regNN.network_predict_v2(dataMat_test, parameters_NN = None, stepPred= True, 
                                           parameters_xNN = regNN_parameters_x, x_norm_parameters = norm_parameters_x, 
                                           parameters_yNN = regNN_parameters_y, y_norm_parameters = norm_parameters_y, 
                                           numSteps=20)
dataMat_regNN_Pred = dataMat_regNN_Pred.T

'''

##########################################################
####  test bike 
'''
### normalizing the data
test_x = np.divide(np.subtract(dataMat_test_bike[:,:-1], norm_parameters_bike['mean']), norm_parameters_bike['stdDev'])
test_x = test_x.T
test_y = dataMat_test_bike[:,-1].T

predictions_test = regNN.network_predict(test_x, parameters_bike, regMode = True, foreAct = 'relu')
print("\ncorrelation coefficients (Test)(regNN): ", np.corrcoef(predictions_test, test_y, rowvar=0)[0,1])
print("\nSquared error (Test)(regNN): ", reg.calError(dataMat_test_bike[:,-1].flatten().A[0].tolist(), predictions_test.A))
reg.showStdLinReg(dataList_test_bike, predictions_test.T)
'''

##########################################################
####  test data4 
'''
predictions_test = regNN.network_predict(train_x_data4, parameters_data4, regMode = True, foreAct = 'relu')
#print("\ncorrelation coefficients (Test)(regNN): ", np.corrcoef(predictions_test, test_y, rowvar=0)[0,1])
#print("\nSquared error (Test)(regNN): ", reg.calError(dataMat_test[:,-1].flatten().A[0].tolist(), predictions_test.A))
reg.showStdLinReg(dataList_data4, predictions_test.T)
'''




############################################################################
#########################       Plot       #################################
#plot.plotTrack_mult(dataMat_org[::5,:])
#plot.plotTrack_mult(dataMat_org[::5,:], dataMat_regNN_Pred, predShow =True, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)
#plot.plot_1234(dataMat_org[::5,:], dataMat_regNN_Pred, origShow=False, predShow =True, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)
#plot.plotTrack_mult(dataMat_org, dataMat_ModPred, predShow =True)
#plot.plot_1234(dataMat_org[::4,:], dataMat_regPred, predShow =True, minX_plot = 0, maxX_plot = 640, minY_plot = 0, maxY_plot = 400)