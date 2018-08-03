import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot
import neural_network as NN
import random




############################################################################
###################  load training and test set   #########################
############################################################################
### street 1234
train_path = 'samples/1234/training_test_set/1234_with_block/';   
test_path  = 'samples/1234/training_test_set/1234_with_block/';  
model_path = "trained_model/2018_08_09/1234_with_block/"
last_steps = "_L2"
file_format = ".dat"
step_size = 2
machine_mode = "classification"


###  load data set for regression mode

training_set_xy_filename = 'training_set_xy_1234Block' + last_steps + file_format
training_set_x_filename = "training_set_x_1234Block" + last_steps + file_format
training_set_y_filename = "training_set_y_1234Block" + last_steps + file_format
test_set_x_filename = "test_set_x_1234Block" + last_steps + file_format
test_set_y_filename = "test_set_y_1234Block" + last_steps + file_format

dataMat_org = read.load_data(train_path + training_set_xy_filename)
dataMat_x_org = read.load_data(train_path + training_set_x_filename)
dataMat_y_org = read.load_data(train_path + training_set_y_filename)
dataMat_x_test = read.load_data(test_path + test_set_x_filename)
dataMat_y_test = read.load_data(test_path + test_set_y_filename)
dataMat_test = np.mat(np.zeros((dataMat_x_test.shape[0],dataMat_x_test.shape[1]+1)))
dataMat_test[:,:-1] = dataMat_x_test
dataMat_test[:,-1] = dataMat_y_test[:,-1]



###  load data set for classification mode

training_posSpeed_set_file = 'training_posSpeed_set_1234Block' + last_steps + file_format
train_destination_set_file = 'training_destination_set_1234Block' + last_steps + file_format
training_posSpeed_set = read.load_data(train_path + training_posSpeed_set_file)
train_destination_set = read.load_data(train_path + train_destination_set_file)

test_posSpeed_set_file = 'test_1234Block_posSpeed_set' + last_steps + file_format
test_destination_set_file = 'test_1234Block_destination_set' + last_steps + file_format
test_posSpeed_set = read.load_data(test_path + test_posSpeed_set_file)
test_destination_set = read.load_data(test_path + test_destination_set_file)




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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
###################  load Tree model   #########################
############################################################################

####################   Regression Tree  #######

#### trajectory
'''
regTreeX_trained_file = 'regTreeX_1234Block_trained_0_5_L1.dat';    regTreeX_trained = reg.loadTree(model_path + regTreeX_trained_file)
regTreeY_trained_file = 'regTreeY_1234Block_trained_0_5_L1.dat';    regTreeY_trained = reg.loadTree(model_path + regTreeY_trained_file)
'''

#### data 4
'''
regTree_trained_file = 'regTree_data4_0_20.dat';    regTree_data4_trained = reg.loadTree(regTree_trained_file)
reg.createPlot(regTree_data4_trained)
'''

####################   Model Tree  #######

#### trajectory
'''
modTreeX_trained_file = 'modTreeX_1234Block_trained_0_20_L1.dat';    modTreeX_trained = reg.loadTree(model_path + modTreeX_trained_file)
modTreeY_trained_file = 'modTreeY_1234Block_trained_0_20_L1.dat';    modTreeY_trained = reg.loadTree(model_path + modTreeY_trained_file)
'''

#### data 4
#modTree_file = 'modTree_data4_0_20.dat';    modTree_data4_trained = reg.loadTree(modTree_file)

#### bike
#modTree_file = 'modTreeX_bike_test_pinv.dat';    modTree_bike_trained = reg.loadTree(modTree_file)



####################   Descent Tree  #######
#### data 
#desTree_file = 'desTree_data4_0_20_min001_epoch30.dat';    desTree_data4_trained = reg.loadTree(desTree_file)

#### bike
#desTree_file = 'desTree_bike_0_20_min001_epoch30.dat';    desTree_bike_trained = reg.loadTree(desTree_file)





############################################################################
#####################   load neural network model   ################## 
############################################################################


##################   load model for trajectory     ##################

#####  regression mode  #####
'''
NN_1234_x_file = 'NN_1234Block_x_trained_v1_L1.dat'; NN_parameters_x = read.load_data(model_path + NN_1234_x_file)
NN_1234_y_file = 'NN_1234Block_y_trained_v1_L1.dat'; NN_parameters_y = read.load_data(model_path + NN_1234_y_file)
norm_parameters_x, train_x = NN.normData(dataMat_x_org[:,:-1]); train_x_target = dataMat_x_org[:,-1].T
norm_parameters_y, train_y = NN.normData(dataMat_y_org[:,:-1]); train_y_target = dataMat_y_org[:,-1].T
'''


#####  classification mode  #####

NN_1234_class_model_file = 'regNN_1234Block_classModel_trained_v2' + last_steps + file_format
NN_1234_class_model_trained = read.load_data(model_path + NN_1234_class_model_file)

## normalizing training data
norm_parameters_x, train_x = NN.normData(training_posSpeed_set)
train_y_destination = train_destination_set.T






#####  load model for bike  #####
'''
regression_network_file =  'NN_bike_testOpt_with_minibatch.dat'
parameters_bike = NN.loadNNetwork(model_path + regression_network_file)
norm_parameters_bike, train_x_bike = NN.normData(dataMat_train_bike[:,:-1])
'''

#####  load model for data4  #####
'''
regression_network_file =  'NN_data4_3.dat'
parameters_data4 = NN.loadNNetwork(regression_network_file)
norm_parameters_data4, train_x_data4 = NN.normData(dataMat_data4[:,:-1])
'''



############################################################################
####################    evaluate Regressino Tree model       #####################
############################################################################

#### evaluate trajectory
'''
## evaluate with train_x ----------------------------------
print('\n\n---------------------   evaluate regression tree   ------------------------------------')
train_accArr_x_regTree = reg.createForeCast_v2(regTreeX_trained, dataMat_x_org[:,:-1])    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (data: train, target: accX, mode: regTree): ", np.corrcoef(train_accArr_x_regTree, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: train, target: accX, mode: regTree): ", reg.cal_MSE(dataMat_x_org[:,-1].flatten().A[0], train_accArr_x_regTree))

train_accArr_Y_regTree = reg.createForeCast_v2(regTreeY_trained, dataMat_y_org[:,:-1])  
print("\ncorrelation coefficients (data: train, target: accY, mode: regTree): ", np.corrcoef(train_accArr_Y_regTree, dataMat_y_org[:,-1], rowvar=0)[0,1])
print("\nMean squared error (data: train, target: accY, mode: regTree): ", reg.cal_MSE(dataMat_y_org[:,-1].flatten().A[0], train_accArr_Y_regTree))


## evaluate with test_x ----------------------------------
test_accArr_x_regTree = reg.createForeCast_v2(regTreeX_trained, dataMat_x_test[:,:-1])    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (data: test, target: accX, mode: regTree): ", np.corrcoef(test_accArr_x_regTree, dataMat_x_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: test, target: accX, mode: regTree): ", reg.cal_MSE(dataMat_x_test[:,-1].flatten().A[0], test_accArr_x_regTree))

test_accArr_Y_regTree = reg.createForeCast_v2(regTreeY_trained, dataMat_y_test[:,:-1])  
print("\ncorrelation coefficients (data: test, target: accY, mode: regTree): ", np.corrcoef(test_accArr_Y_regTree, dataMat_y_test[:,-1], rowvar=0)[0,1])
print("\nMean squared error (data: test, target: accY, mode: regTree): ", reg.cal_MSE(dataMat_y_test[:,-1].flatten().A[0], test_accArr_Y_regTree))
'''



#### evaulate data4
'''
yArr_pred_test = reg.createForeCast(regTree_data4_trained, dataMat_data4[:,0])
reg.showTree(dataList_data4, yArr_pred_test, mode = "regTree")
'''


############################################################################
####################    evaluate Model Tree model     ##########################
############################################################################

#### evaluate trajectory
'''
## evaluate with train_x ----------------------------------
print('\n\n---------------------   evaluate model tree   ------------------------------------')
train_accArr_x_modTree = reg.createForeCast_v2(modTreeX_trained, dataMat_x_org[:,:-1], reg.modelTreeEval)    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (data: train, target: accX, mode: modTree): ", np.corrcoef(train_accArr_x_modTree, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: train, target: accX, mode: modTree): ", reg.cal_MSE(dataMat_x_org[:,-1].flatten().A[0], train_accArr_x_modTree))

train_accArr_Y_modTree = reg.createForeCast_v2(modTreeY_trained, dataMat_y_org[:,:-1], reg.modelTreeEval)  
print("\ncorrelation coefficients (data: train, target: accY, mode: modTree): ", np.corrcoef(train_accArr_Y_modTree, dataMat_y_org[:,-1], rowvar=0)[0,1])
print("\nMean squared error (data: train, target: accY, mode: modTree): ", reg.cal_MSE(dataMat_y_org[:,-1].flatten().A[0], train_accArr_Y_modTree))


## evaluate with test_x ----------------------------------
test_accArr_x_modTree = reg.createForeCast_v2(modTreeX_trained, dataMat_x_test[:,:-1], reg.modelTreeEval)    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (data: test, target: accX, mode: modTree): ", np.corrcoef(test_accArr_x_modTree, dataMat_x_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: test, target: accX, mode: modTree): ", reg.cal_MSE(dataMat_x_test[:,-1].flatten().A[0], test_accArr_x_modTree))

test_accArr_Y_modTree = reg.createForeCast_v2(modTreeY_trained, dataMat_y_test[:,:-1], reg.modelTreeEval)  
print("\ncorrelation coefficients (data: test, target: accY, mode: modTree): ", np.corrcoef(test_accArr_Y_modTree, dataMat_y_test[:,-1], rowvar=0)[0,1])
print("\nMean squared error (data: test, target: accY, mode: modTree): ", reg.cal_MSE(dataMat_y_test[:,-1].flatten().A[0], test_accArr_Y_modTree))
'''


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
####################    evaluate nerual network model      #################
############################################################################

####  evaulate trajectory 

#############   test regression   #############
'''
## evaluate with train_x ----------------------------------
print('\n\n---------------------   evaluate neural network model for regression  ------------------------------------')
train_accArr_x_NN = NN.network_predict(train_x, NN_parameters_x, regMode = True, foreAct = 'relu')
train_accArr_x_NN = np.array(train_accArr_x_NN).ravel()  # convert matrix to array !!
print("\ncorrelation coefficients (data: train, target: accX, mode: NN): ", np.corrcoef(train_accArr_x_NN, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: train, target: accX, mode: NN): ", reg.cal_MSE(dataMat_x_org[:,-1].flatten().A[0], train_accArr_x_NN))

train_accArr_y_NN = NN.network_predict(train_y, NN_parameters_y, regMode = True, foreAct = 'relu')
train_accArr_y_NN = np.array(train_accArr_y_NN).ravel()
print("\ncorrelation coefficients (data: train, target: accY, mode: NN): ", np.corrcoef(train_accArr_y_NN, dataMat_y_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: train, target: accY, mode: NN): ", reg.cal_MSE(dataMat_y_org[:,-1].flatten().A[0], train_accArr_y_NN))


## evaluate with test_x ----------------------------------  normalization !!
dataMat_x_test_NN = dataMat_x_test[:,:-1].T 
dataMat_y_test_NN = dataMat_y_test[:,:-1].T 

test_accArr_x_NN = NN.network_predict(dataMat_x_test_NN, NN_parameters_x, norm_parameters_NN = norm_parameters_x,  regMode = True, foreAct = 'relu')
test_accArr_x_NN = np.array(test_accArr_x_NN).ravel()  # convert matrix to array !!
print("\ncorrelation coefficients (data: test, target: accX, mode: NN): ", np.corrcoef(test_accArr_x_NN, dataMat_x_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: test, target: accX, mode: NN): ", reg.cal_MSE(dataMat_x_test[:,-1].flatten().A[0], test_accArr_x_NN))

test_accArr_y_NN = NN.network_predict(dataMat_y_test_NN, NN_parameters_y, norm_parameters_NN = norm_parameters_y, regMode = True, foreAct = 'relu')
test_accArr_y_NN = np.array(test_accArr_y_NN).ravel()
print("\ncorrelation coefficients (data: test, target: accY, mode: NN): ", np.corrcoef(test_accArr_y_NN, dataMat_y_test[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nMean squared error (data: test, target: accY, mode: NN): ", reg.cal_MSE(dataMat_y_test[:,-1].flatten().A[0], test_accArr_y_NN))
'''



#############   test classification   #############
## evaluate with train_x ----------------------------------
'''
print('\n\n---------------------   evaluate neural network model for classification   ------------------------------------')

train_destination_predNN = NN.network_predict_v3(train_x, NN_1234_class_model_trained, prediction_mode = "classification", foreAct = 'relu')
train_destination_prediction, train_error_list = NN.prediction_destination(train_destination_predNN, train_y_destination)
print("\ncorrect recognition rate (data: train, target: destination, mode: NN): ", train_destination_prediction)    #  dataMat_x[:,-1]: transfer the acceleration !


test_destination_predNN = NN.network_predict_v3(test_posSpeed_set.T, NN_1234_class_model_trained, norm_parameters_NN = norm_parameters_x, prediction_mode = "classification", foreAct = 'relu')
test_destination_prediction, test_error_list = NN.prediction_destination(test_destination_predNN, test_destination_set.T)
print("\ncorrect recognition rate (data: test, target: destination, mode: NN): ", test_destination_prediction)    #  dataMat_x[:,-1]: transfer the acceleration !
'''

## show error points
#plot.plot_1234_v3(dataMat_org[::4,:], dataMat_error=training_posSpeed_set[train_error_list,:],  origShow=False, predShow =True, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)


############################################################################
#########################       Test       #################################
############################################################################


#### use original data
'''
random_pos = random.randint(0, dataMat_y_test.shape[0]-1)
dataMat_test = dataMat_y_test[random_pos,0:-1]           # make the starting point as test data
'''

#### customize the test data

#dataMat_test = np.mat(np.array([320, 290, 0, 0]))


#dataMat_test = np.mat(np.array([330, 270, -120, 100,    320,290,0,0])) # 222





#dataMat_test = np.mat(np.array([270,310,-20,-10, 290,290,10,-10,    320,270,10,-10]))  # 333

############################################################################
############################################################################
#### test the Regression Tree !!

#dataMat_regTree_Pred = reg.createForeCast_v2(regTreeX_trained, dataMat_test, modelEval = reg.regTreeEval, stepPred=True, 
#                                     treeX_trained=regTreeX_trained, treeY_trained=regTreeY_trained, numSteps=15)


############################################################################
############################################################################
#### test the Model Tree !!
#dataMat_modTree_Pred = reg.createForeCast_v2(modTreeX_trained, dataMat_test, modelEval = reg.modelTreeEval, stepPred=True, 
 #                                    treeX_trained=modTreeX_trained, treeY_trained=modTreeY_trained, numSteps=15)





############################################################################
############################################################################
#### test the neural network !!      
####  test walk trace 

#############   test regression   #############
 
## for the neural network, features are rows vectors and samples are column vectors
'''
dataMat_test = dataMat_test.T 
## predict!
dataMat_NN_Pred = NN.network_predict_v2(dataMat_test, parameters_NN = None, stepPred= True, 
                                           parameters_xNN = NN_parameters_x, x_norm_parameters = norm_parameters_x, 
                                           parameters_yNN = NN_parameters_y, y_norm_parameters = norm_parameters_y, 
                                           numSteps=15)
dataMat_NN_Pred = dataMat_NN_Pred.T

'''

#############   test classification   #############

test_destination_NN = NN.network_predict_v3(test_posSpeed_set.T, NN_1234_class_model_trained, norm_parameters_NN = norm_parameters_x, prediction_mode = "classification", foreAct = 'relu')
print(test_destination_NN)





##########################################################
####  test bike 
'''
### normalizing the data
test_x = np.divide(np.subtract(dataMat_test_bike[:,:-1], norm_parameters_bike['mean']), norm_parameters_bike['stdDev'])
test_x = test_x.T
test_y = dataMat_test_bike[:,-1].T

predictions_test = NN.network_predict(test_x, parameters_bike, regMode = True, foreAct = 'relu')
print("\ncorrelation coefficients (Test)(NN): ", np.corrcoef(predictions_test, test_y, rowvar=0)[0,1])
print("\nSquared error (Test)(NN): ", reg.calError(dataMat_test_bike[:,-1].flatten().A[0].tolist(), predictions_test.A))
reg.showStdLinReg(dataList_test_bike, predictions_test.T)
'''

##########################################################
####  test data4 
'''
predictions_test = NN.network_predict(train_x_data4, parameters_data4, regMode = True, foreAct = 'relu')
#print("\ncorrelation coefficients (Test)(NN): ", np.corrcoef(predictions_test, test_y, rowvar=0)[0,1])
#print("\nSquared error (Test)(NN): ", reg.calError(dataMat_test[:,-1].flatten().A[0].tolist(), predictions_test.A))
reg.showStdLinReg(dataList_data4, predictions_test.T)
'''




############################################################################
#########################       Plot       #################################
#plot.plot_1234_v3(dataMat_org[::5,:], dataMat_regTree_Pred=dataMat_regTree_Pred, dataMat_modTree_Pred=dataMat_modTree_Pred, dataMat_NN_Pred=dataMat_NN_Pred, origShow=False, predShow =True, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)
#plot.plot_1234_v3(dataMat_org[::5,:], dataMat_regTree_Pred=dataMat_regTree_Pred,  origShow=False, predShow =True, minX_plot = 0, maxX_plot = 650, minY_plot = 0, maxY_plot = 400)

#plot.plotTrack_mult(dataMat_org, dataMat_ModPred, predShow =True)
#plot.plot_1234(dataMat_org[::4,:], dataMat_regPred, predShow =True, minX_plot = 0, maxX_plot = 640, minY_plot = 0, maxY_plot = 400)