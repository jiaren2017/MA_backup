import numpy as np
import h5py
import pandas as pd

def read_parameter(group, sub):
    sub_group = group.get(sub)   
    position = np.array(sub_group.get('Positions'))
    velocity = np.array(sub_group.get('Velocity'))
    accerlation = np.array(sub_group.get('Acceleration'))
    return accerlation, position, velocity


def read_csv(Filename):
    data = pd.read_csv(Filename)
    print(data)


def array_to_matrix(accArr, posArr, speedArr):
    posMat    = np.mat(posArr)
    speedMat  = np.mat(speedArr)
    accMat    = np.mat(accArr)
    dataMat   = np.mat(np.ones((len(accArr),6)))         # pos(x,y) + speed(x,y) + acc(x,y)
    dataMat[:,0:2] = posMat[1:-1,:]   # assign the position matrix, WITHOUT! the FIRST and LAST position
    dataMat[:,2:4] = speedMat[:-1,:]  # assign the speed matrix,    WITHOUT! the LAST speed
    dataMat[:,4:]  = accMat[:,:]      # assign the accelerate matrix 
    return dataMat


def read_hdf5_auto(path, files):
    dataMat_org = np.mat([0,0,0,0,0,0])              # init. a zero matrix:  pos(x,y) + speed(x,y) + acc(x,y)
    for subFile in list(files):                      # populate all files ['file_1', 'file_2'...]
        h5_file = h5py.File(path + subFile)
#        print('Filename is: \n', path + subFile)
        
        for subKey in list(h5_file):                    # ['key_1', 'key_2'...]
#            print('current key: \n', subKey)
            group = h5_file.get(subKey)
#            print('\ncurrent group: \n', group)
            for subGroup in list(group):            # ['1', '2',...]
                accArr, posArr, speedArr = read_parameter(group, subGroup)
                dataMat_current = array_to_matrix(accArr, posArr, speedArr)
#                print('\ndataMat_current is: \n', dataMat_current)
                dataMat_org = np.row_stack((dataMat_org, dataMat_current))   # combine two matrix
#                print('\ndataMat_org is: \n', dataMat_org)
            
    dataMat_x_org = np.mat(np.ones((len(dataMat_org)-1,5)))             # pos(x,y) + speed(x,y) + acc(x)
    dataMat_y_org = np.mat(np.ones((len(dataMat_org)-1,5)))             # pos(x,y) + speed(x,y) + acc(y)
    dataMat_x_org = dataMat_org[1:,:-1]                                 # generate the matrix for training x-accelerate
    dataMat_y_org[:,:-1] = dataMat_org[1:,:-2]; dataMat_y_org[:,4] = dataMat_org[1:,5] # generate the matrix for training y-accelerate
    
    return dataMat_org[1:,:], dataMat_x_org, dataMat_y_org       # return the matrix, without the first intial row [0,0,0,0,0,0]
    


def read_hdf5_auto_v2(path, files, train_step_size = 1):
    #  if for example train_step_size is 2, dataMat_org:  pos(x1,y1) + speed(x1,y1) + pos(x2,y2) + speed(x2,y2) + acc(x2,y2)
    dataMat_org = np.mat(np.zeros((1, train_step_size*4+2)))   # init. a zero matrix, 
#    print('dataMat_org is: \n', dataMat_org)
    
    for subFile in list(files):                     # populate all files ['file_1', 'file_2'...]
        h5_file = h5py.File(path + subFile)
#        print('Filename is: \n', path + subFile)
        
        for subKey in list(h5_file):                # ['key_1', 'key_2'...]
#            print('current key: \n', subKey)
            group = h5_file.get(subKey)
#            print('\ncurrent group: \n', group)
            for subGroup in list(group):            # ['1', '2',...]
#                print('\nsubGroup: \n', subGroup)
                accArr, posArr, speedArr = read_parameter(group, subGroup)
#                print('\naccArr, posArr, speedArr: \n', accArr, posArr, speedArr)
                dataMat_current = array_to_matrix(accArr, posArr, speedArr)
#                print('\ndataMat_current is: \n', dataMat_current)
#                print('\ndataMat_current[0] is: \n', dataMat_current[0])
#                print('\nlength of dataMat_current is: \n', dataMat_current.shape[0])
                i = 0
                while i <= (dataMat_current.shape[0] - train_step_size):
                    dataMat_new = np.mat(np.zeros((1, train_step_size*4+2)))    # init. a zero matrix, 
                    j = 0
                    while j < train_step_size:      # assign the pos(x,y) + speed(x,y)
                        dataMat_new[0, j*4:(j+1)*4] = dataMat_current[i+j,0:4]
                        j = j + 1
                    dataMat_new[0, -2:] = dataMat_current[i+j-1,-2:]             # assign the acc(x,y)
                    dataMat_org = np.row_stack((dataMat_org, dataMat_new))      # combine two matrix
                    i = i + 1
#           print('\ndataMat_org is: \n', dataMat_org)
      
    dataMat_x_org = np.mat(np.ones((dataMat_org.shape[0]-1, dataMat_org.shape[1]-1)))                   # pos(x,y) + speed(x,y) + acc(x)
    dataMat_y_org = np.mat(np.ones((dataMat_org.shape[0]-1, dataMat_org.shape[1]-1)))                   # pos(x,y) + speed(x,y) + acc(y)
    dataMat_x_org = dataMat_org[1:,:-1]                                                     # generate the matrix for training x-accelerate
    dataMat_y_org[:,:-1] = dataMat_org[1:,:-2]; dataMat_y_org[:,-1] = dataMat_org[1:,-1]    # generate the matrix for training y-accelerate
    
    return dataMat_org[1:,:], dataMat_x_org, dataMat_y_org       # return the matrix, without the first intial row [0,0,0,0,0,0]

###################################################### 
'''
path = 'samples/'
Filename = ['lin_zero_read_test.hdf5','cub_400.hdf5']
dataMat_org, dataMat_x_org, dataMat_y_org = read_hdf5_auto(path, Filename)
'''

