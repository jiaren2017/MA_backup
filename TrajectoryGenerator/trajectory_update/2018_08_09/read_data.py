import numpy as np
import h5py
import pandas as pd
import pickle

##########################   save and load data    ##############################
def save_data(inputData, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputData, fw)
    
def load_data(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
    
    
##########################   read data    ##############################

def read_parameter(group, sub):
    sub_group = group.get(sub)   
    position = np.array(sub_group.get('Positions'))
    velocity = np.array(sub_group.get('Velocity'))
    accerlation = np.array(sub_group.get('Acceleration'))
    return position, velocity, accerlation


def read_csv(Filename):
    data = pd.read_csv(Filename)
    #print(data)


def array_to_matrix(accArr, posArr, speedArr, mode = "regression"):
    if (mode == "regression"):
        posMat    = np.mat(posArr)
        speedMat  = np.mat(speedArr)
        accMat    = np.mat(accArr)
        dataMat   = np.mat(np.ones((len(accArr),6)))         # pos(x,y) + speed(x,y) + acc(x,y)
        dataMat[:,0:2] = posMat[1:-1,:]   # assign the position matrix, WITHOUT! the FIRST and LAST position
        dataMat[:,2:4] = speedMat[:-1,:]  # assign the speed matrix,    WITHOUT! the LAST speed
        dataMat[:,4:]  = accMat[:,:]      # assign the accelerate matrix 
    elif (mode == "classification"):
        posMat    = np.mat(posArr)
        speedMat  = np.mat(speedArr)
        dataMat   = np.mat(np.ones((len(speedArr),4)))         # ONLY!!  pos(x,y) + speed(x,y)  
        dataMat[:,0:2] = posMat[1:,:]   # assign the position matrix, WITHOUT! the FIRST and LAST position
        dataMat[:,2:4] = speedMat[:,:]  # assign the speed matrix,    WITHOUT! the LAST speed
    return dataMat


def destination_analyse(pos_x, pos_y, map="1234"):
    if map == '1234':
        if(pos_x < 100):
            if(pos_y < 100):     return 1
            elif(pos_y >250):    return 4
        elif(pos_x > 500):
            if(pos_y < 100):     return 2
            elif(pos_y >250):    return 3
        else:
            raise NameError("No valid destination detected!!")
            
    elif map == 'forum':
        if(pos_x <= 180) and (pos_y >= 410 ):                       return 1  ## main entry   [0,410], [180,410], [180, 480], [0, 480]
        elif(pos_x <= 60) and (pos_y <= 300 ):                      return 2  ## Auditorium   [0,0], [60,0], [60,300], [0,300]
        elif(pos_x >= 60) and (pos_x <= 215 ) and (pos_y <= 60 ):    return 3  ## Cafe,        [60,0], [215,0], [215,60], [60,60]
        elif(pos_x >= 215) and (pos_x <= 330 ) and (pos_y <= 60 ):   return 4  ## Stairs,      [215,0], [330,0], [330,60], [215,60]
        elif(pos_x >= 445) and (pos_x <= 570 ) and (pos_y <= 60 ):   return 5  ## Elevator,        [445,0], [570,0], [570,60], [445,60]
        elif(pos_x >= 570) and (pos_y <= 60 ):                      return 6  ## night exit door: [570,0], [640,0], [640,60], [570,60]
        elif(pos_x >= 570) and (pos_y >= 60 ) and (pos_y <= 420 ):   return 7  ## robot lab:       [570,60], [640,60], [640, 420], [570, 420]
        elif(pos_x >= 550) and (pos_y >= 410 ):                     return 8  ## vision lab:      [550,420], [640,410], [640, 480], [550, 480]
        elif(pos_x >= 200) and (pos_x <= 300 ) and (pos_y >= 410 ):  return 9  ## reception:       [200,420], [300,410], [300, 480], [200, 480]
        else:
            print("\npos_x: ", pos_x)
            print("\npos_y: ", pos_y)
            raise NameError("\nNo valid destination detected!!")

            
def read_hdf5_auto_v3(path, files, train_step_size = 1, mode = "regression"):
    if (mode == "regression"):    ## prepare the data for training regression: pos(x,y) + speed(x,y) + acc(x2,y2)
    
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
                    posArr, speedArr, accArr = read_parameter(group, subGroup)
                    print('\nposArr: \n', posArr)
    #                print('\nLast posArr: \n', posArr[-1])
                    print('\nspeedArr: \n', speedArr)
                    print('\naccArr: \n', accArr)
                    dataMat_current = array_to_matrix(accArr, posArr, speedArr, mode)
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
        dataMat_x_org = dataMat_org[1:,:-1]                                                     # generate the matrix for training x-accelerate, without the first zero vector
        dataMat_y_org[:,:-1] = dataMat_org[1:,:-2]; dataMat_y_org[:,-1] = dataMat_org[1:,-1]    # generate the matrix for training y-accelerate, without the first zero vector
        return dataMat_org[1:,:], dataMat_x_org, dataMat_y_org       # return the matrix, without the first intial row [0,0,0,0,0,0]

    elif(mode == "classification"):    ## prepare the data for training classification:
        dataMat_traing_set_x = np.mat(np.zeros((1, train_step_size*4)))   # init. a zero matrix for training input set:  pos(x,y) + speed(x,y) 
        dataMat_traing_set_y = np.mat(np.zeros((1, 4)))   # init. a zero matrix for training input set:  class(destination)
        for subFile in list(files):                     # populate all files ['file_1', 'file_2'...]
            h5_file = h5py.File(path + subFile)
    #        print('Filename is: \n', path + subFile)
            
            for subKey in list(h5_file):                # ['key_1', 'key_2'...]
    #            print('current key: \n', subKey)
                group = h5_file.get(subKey)
    #            print('\ncurrent group: \n', group)
                for subGroup in list(group):            # ['1', '2',...]
    #                print('\nsubGroup: \n', subGroup)
                    posArr, speedArr, accArr = read_parameter(group, subGroup)
                    dataMat_current = array_to_matrix(accArr, posArr, speedArr, mode)
    #                print('\ndataMat_current is: \n', dataMat_current)
                    
                    ## build the dataMat_traing_set_x:    pos(x,y) + speed(x,y) 
                    i = 0
                    while i <= (dataMat_current.shape[0] - train_step_size):
                        dataMat_x_temp = np.mat(np.zeros((1, train_step_size*4)))    # init. a zero matrix, 
                        j = 0
                        while j < train_step_size:      # assign the pos(x,y) + speed(x,y)
                            dataMat_x_temp[0, j*4:(j+1)*4] = dataMat_current[i+j,0:4]
                            j = j + 1
                        dataMat_traing_set_x = np.row_stack((dataMat_traing_set_x, dataMat_x_temp))      # combine two matrix
                        i = i + 1
                        
                    ## calculate the class of destination    
                    dataMat_y_temp = np.mat(np.zeros((i, 4)))        # init. a zero matrix, 
                    destination_temp = destination_analyse(dataMat_current[-1,0], dataMat_current[-1,1])  # detect the destination based on the last position!
                    dataMat_y_temp[:,destination_temp-1] = 1    
                    dataMat_traing_set_y = np.row_stack((dataMat_traing_set_y, dataMat_y_temp))     # combine two matrix   
                    
        #print('\ndataMat_traing_set_x is: \n', dataMat_traing_set_x)
        #print('\ndataMat_traing_set_y is: \n', dataMat_traing_set_y)
        return dataMat_traing_set_x[1:,:], dataMat_traing_set_y[1:,:]                               # return the training set x (pos and speed) and training set y (destination)            

    
def shuffle_data(dataMat_x, dataMat_y):                  # row vector: samples,  column vector: features
#    np.random.seed(10)            # To make your "random" minibatches the same as ours
    m = dataMat_x.shape[0]                    # number of training examples
    permutation = list(np.random.permutation(m))
    return dataMat_x[permutation, :], dataMat_y[permutation, :]





    
##########################   load Forum trajectory set in 'txt' format    ##############################
def load_forum_trajectory_set(path, files, train_step_size = 1, mode = "regression", minStepSize=4, maxStepSize=7):

    removed_steps_list = []
    removed_lines_list = []
    
    if (mode == "regression"):    ## prepare the data for training regression: pos(x,y) + speed(x,y) + acc(x2,y2)
        #  if for example train_step_size is 2, dataMat_org:  pos(x1,y1) + speed(x1,y1) + pos(x2,y2) + speed(x2,y2) + acc(x2,y2)
        dataMat_org = np.mat(np.zeros((1, train_step_size*4+2)))   # init. a zero matrix, 
        destination_set = np.mat(np.zeros((1, train_step_size*4+2)))
        
        for subFile in list(files):                     # populate all files ['file_1', 'file_2'...]
            with open(path + subFile, 'r') as f:
                for line in f.readlines():
                    line = line.strip()              ## delete all spaces in the begin and end of file
                    if line[0:5] == 'TRACK':
                        pos_x_list, pos_y_list = get_pos_in_list(line)
                        pos_x_post_list, pos_y_post_list, removed_steps_list_temp = remove_steps(pos_x_list, pos_y_list, min_step_size = minStepSize, max_step_size = maxStepSize)
                        if pos_x_post_list == None:
                            removed_lines_list.append(line[7])
                            continue
                        
                        removed_steps_list.append(removed_steps_list_temp)
                        pos_arr, speed_arr, acc_arr = gen_pos_speed_acc_array(pos_x_post_list, pos_y_post_list)
                        dataMat_current = array_to_matrix(acc_arr, pos_arr, speed_arr, mode)    ## dataMat_current = pos(x,y) + speed(x,y) + acc(x,y) or pos(x,y) + speed(x,y)
                        destination_set = np.row_stack((destination_set, dataMat_current[-1,:]))      # append the destination of each trajectory
                        
                        ## build dataMat_new according to the format L1, L2 or L3
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
                            
        dataMat_x_org = np.mat(np.ones((dataMat_org.shape[0]-1, dataMat_org.shape[1]-1)))                   # pos(x,y) + speed(x,y) + acc(x)
        dataMat_y_org = np.mat(np.ones((dataMat_org.shape[0]-1, dataMat_org.shape[1]-1)))                   # pos(x,y) + speed(x,y) + acc(y)
        dataMat_x_org = dataMat_org[1:,:-1]                                                     # generate the matrix for training x-accelerate, without the first zero vector
        dataMat_y_org[:,:-1] = dataMat_org[1:,:-2]; dataMat_y_org[:,-1] = dataMat_org[1:,-1]    # generate the matrix for training y-accelerate, without the first zero vector
        return dataMat_org[1:,:], dataMat_x_org, dataMat_y_org, removed_steps_list, removed_lines_list, destination_set[1:,:]       # return the matrix, without the first intial row [0,0,0,0,0,0]

    elif(mode == "classification"):    ## prepare the data for training classification:
        dataMat_traing_set_x = np.mat(np.zeros((1, train_step_size*4)))   # init. a zero matrix for training input set:  pos(x,y) + speed(x,y) 
        dataMat_traing_set_y = np.mat(np.zeros((1, 9)))   # init. a zero matrix for training input set:  class(destination)
        
        for subFile in list(files):                     # populate all files ['file_1', 'file_2'...]
            with open(path + subFile, 'r') as f:
                for line in f.readlines():
                    line = line.strip()              ## delete all spaces in the begin and end of file
                    if line[0:5] == 'TRACK':
                        pos_x_list, pos_y_list = get_pos_in_list(line)
                        pos_x_post_list, pos_y_post_list, removed_steps_list_temp = remove_steps(pos_x_list, pos_y_list, min_step_size = minStepSize, max_step_size = maxStepSize)
                        if pos_x_post_list == None:     ## skip the line removed and try next one with 'continue'
                            removed_lines_list.append(line[7])
                            continue
                        
                        removed_steps_list.append(removed_steps_list_temp)
                        pos_arr, speed_arr, acc_arr = gen_pos_speed_acc_array(pos_x_post_list, pos_y_post_list)
                        dataMat_current = array_to_matrix(acc_arr, pos_arr, speed_arr, mode)
        #                print('\ndataMat_current is: \n', dataMat_current)
                        
                        ## build the dataMat_traing_set_x:    pos(x,y) + speed(x,y) 
                        i = 0
                        while i <= (dataMat_current.shape[0] - train_step_size):
                            dataMat_x_temp = np.mat(np.zeros((1, train_step_size*4)))    # init. a zero matrix, 
                            j = 0
                            while j < train_step_size:      # assign the pos(x,y) + speed(x,y)
                                dataMat_x_temp[0, j*4:(j+1)*4] = dataMat_current[i+j,0:4]
                                j = j + 1
                            dataMat_traing_set_x = np.row_stack((dataMat_traing_set_x, dataMat_x_temp))      # combine two matrix
                            i = i + 1
                            
                        ## calculate the class of destination    
                        dataMat_y_temp = np.mat(np.zeros((i, 9)))        # init. a zero matrix, 
                        destination_temp = destination_analyse(dataMat_current[-1,0], dataMat_current[-1,1], map='forum')  # detect the destination based on the last position!
                        dataMat_y_temp[:,destination_temp-1] = 1    
                        dataMat_traing_set_y = np.row_stack((dataMat_traing_set_y, dataMat_y_temp))     # combine two matrix   
                    
        #print('\ndataMat_traing_set_x is: \n', dataMat_traing_set_x)
        #print('\ndataMat_traing_set_y is: \n', dataMat_traing_set_y)
        return dataMat_traing_set_x[1:,:], dataMat_traing_set_y[1:,:], removed_steps_list, removed_lines_list   # return the training set x (pos and speed) and training set y (destination)            

                
                
def get_pos_in_list(single_line):
    pos_mat = np.mat(np.zeros((1,2)))
 
    line = single_line[single_line.find('[[')+2:-3]     ## find and cut the "[[" in the begin and the "]];" in the end
    str = ''.join(line)         ## turn list to string    
#    print("\nwhole line is: ", str)
    
    line = str.split('[')       ## cut the "["
    str = ''.join(line)         ## turn list to string    
#    print("\ncut the \"[\": ", str)
    
    line = str.split(']')       ## cut the "]"
    str = ''.join(line)         ## turn list to string    
#    print("\ncut the \"]\": ", str)
    
    line = str.replace(';', ' ')        ## replace all ';' with space
    str = ''.join(line)                 ## turn list to string  
#    print("\nreplace \";\" with space: ", str+'test')
    
    line = str.split()                  ## split the whole list with space like ['a', 'b', 'c' ...  ]
    assert(len(line)%3 == 0)
    
    pos_x_list = list(map(eval, line[::3]))             ## convert the string to int/float
    pos_y_list = list(map(eval, line[1:][::3]))         ## convert the string to int/float
#    print("\nx-value: ", pos_x_list)
#    print("\ny-value: ", pos_y_list)
    return pos_x_list, pos_y_list
    

def remove_steps(pos_x_list, pos_y_list, min_step_size=1, max_step_size=7):
    pos_x_post_list = pos_x_list
    pos_y_post_list = pos_y_list
    removed_steps_list = []

    ## find all the same steps
    for i in range(1, len(pos_x_post_list)-1):
#        if(np.abs(pos_x_post_list[i] - pos_x_post_list[i-1]) < mind_step_size) and (np.abs(pos_y_post_list[i] - pos_y_post_list[i-1]) < mind_step_size):
        if np.sqrt(np.square(pos_x_post_list[i] - pos_x_post_list[i-1]) + np.square(pos_y_post_list[i] - pos_y_post_list[i-1])) >= max_step_size:
            return None,None,None
            
        #if(np.abs(pos_x_post_list[i] - pos_x_post_list[i-1]) < min_step_size) and (np.abs(pos_y_post_list[i] - pos_y_post_list[i-1]) < min_step_size):
        if np.sqrt(np.square(pos_x_post_list[i] - pos_x_post_list[i-1]) + np.square(pos_y_post_list[i] - pos_y_post_list[i-1])) < min_step_size:
            removed_steps_list.append(i)
            
    ## delete the same steps        
    i = 0
    for j in removed_steps_list:
        pos_x_post_list.pop(j-i)
        pos_y_post_list.pop(j-i)    
        i += 1
    
#    print("\npos_x_post_list: ", pos_x_post_list)
#    print("\npos_y_post_list: ", pos_y_post_list)
#    print("\nsame_steps_list: ", same_steps_list)
    return pos_x_post_list, pos_y_post_list, removed_steps_list
    
    
def gen_pos_speed_acc_array(pos_x_arr, pos_y_arr):
    pos_arr = np.zeros((len(pos_x_arr), 2))
    pos_arr[:,0] = pos_x_arr;   
    pos_arr[:,1] = pos_y_arr
    
    speed_arr = np.zeros((len(pos_x_arr)-1, 2))
    speed_arr[:,0] = pos_arr[1:,0] - pos_arr[:-1,0];     
    speed_arr[:,1] = pos_arr[1:,1] - pos_arr[:-1,1]
    
    acc_arr = np.zeros((len(pos_x_arr)-2, 2))
    acc_arr[:,0] = speed_arr[1:,0] - speed_arr[:-1,0];     
    acc_arr[:,1] = speed_arr[1:,1] - speed_arr[:-1,1]
    
#    print("\npos_arr: \n", pos_arr)
#    print("\nspeed_arr: \n", speed_arr)
#    print("\nacc_arr: \n", acc_arr)
    return pos_arr, speed_arr, acc_arr
    

def cal_mean_distance(mat_x, mat_y):
    return np.mean(np.sqrt(np.square(mat_x) + np.square(mat_y)))

    
def cal_rms(dataMat):
    return np.sqrt(np.mean(np.square(dataMat)))
    
    
def build_destination_set(path, files, last_steps = 1):


    dest_x_list = []; dest_y_list = []
    for subFile in list(files):                     # populate all files ['file_1', 'file_2'...]
                with open(path + subFile, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()              ## delete all spaces in the begin and end of file
                        if line[0:5] == 'TRACK':
                            pos_x_list, pos_y_list = get_pos_in_list(line)
                            dest_x_list.append(pos_x_list[-last_steps:])  
                            dest_y_list.append(pos_y_list[-last_steps:]) 
    
    destination_set = np.mat(np.zeros((len(dest_x_list), 2)))
    destination_set[:,0] = dest_x_list
    destination_set[:,1] = dest_y_list
    return destination_set
    
    
    
    
    
                            


   
###################################################### 

# TRACK.R4=[[0 2 2204];[1 2 2204];[1 2 2205];[2 4 2276];[3 5 2276];[3 5 2276];[6 7 2276]];

'''
path = 'samples/'
Filename = ['lin_zero_read_test.hdf5','cub_400.hdf5']
dataMat_org, dataMat_x_org, dataMat_y_org = read_hdf5_auto(path, Filename)
'''

'''
% Total number of trajectories in file are  700 

Properties.R1=[98 60 158 560.65 26.82 30.98 197.30 9.77 0.00 0.00 29.26 59.37 0.46 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.18 0.03 0.00 0.00 11.07 73.04 5.99 0.00 0.08 10.35 18.70 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 1.92 0.01 0.00 0.00 9.33 65.94 1.80 0.00 0.00 6.49 12.16 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 1.71 0.00 0.00 0.00 5.90 37.78 ];
 TRACK.R1=[[468 8 60];[468 8 61]];
Properties.R2=[89 358 446 446.64 26.25 27.57 104.78 42.12 0.01 0.00 9.38 109.18 3.55 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.17 0.03 0.00 0.00 3.65 39.81 10.02 0.00 0.08 4.15 17.64 1.22 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.16 0.01 0.00 0.00 1.99 41.64 9.74 0.00 0.00 2.65 15.30 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.09 0.02 0.00 0.00 0.45 28.78 ];
 TRACK.R2=[[119 447 358];[123 444 359];[127 440 360]];
'''





