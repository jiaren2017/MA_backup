import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch

def getPosArr(dataMat):                                     # extract the position data from matrix
    if (np.shape(dataMat)[1] > 4):                          # more than one starting step,  dataMat: pos(x,y) + speed(x,y) + pos(x,y) + speed(x,y) + ...
        posCount_per_line = np.shape(dataMat)[1] // 4
        posArr = np.zeros((np.shape(dataMat)[0] + posCount_per_line - 1, 2))
        posArr[:np.shape(dataMat)[0],0] = np.array(dataMat[:, 0]).ravel()       # pos x
        posArr[:np.shape(dataMat)[0],1] = np.array(dataMat[:, 1]).ravel()       # pos y
        for i in range(1, posCount_per_line):
            posArr[-posCount_per_line+i, 0] = dataMat[-1, i*4]                  # pos x
            posArr[-posCount_per_line+i, 1] = dataMat[-1, i*4+1]                # pos y
            
    else:                                                   # only one starting step, dataMat: pos(x,y) + speed(x,y)
        posArr = np.zeros((np.shape(dataMat)[0],2))
        posArr[:,0] = np.array(dataMat[:,0]).ravel()
        posArr[:,1] = np.array(dataMat[:,1]).ravel()
    
    return posArr


def plotTrack_single(dataMat_org, dataMat_Pred=None, predShow =False, fig_size = (10,10)):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data
    plt.figure(1, figsize=fig_size)
#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")            # set begiing and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis

    for i in range(np.shape(posArr_org)[0]-1):     # plot the original route
        ax.annotate("",
                    xy=posArr_org[i+1], xycoords='data',        # xy: goal
                    xytext=posArr_org[i], textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3"))
         
    ax.scatter(posArr_org[0][0], posArr_org[0][1], s = 150, c = 'green',alpha = 1, marker='s')       # plot the beginning point
    ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 150, c = 'green',alpha = 1, marker='o')      # plot the ending point
    
    
    
    if (predShow == True):                   # plot the predicted route   
        posArr_Pred = getPosArr(dataMat_Pred)
        for i in range(np.shape(posArr_Pred)[0]-1):     
            ax.annotate("",
                        xy=posArr_Pred[i+1], xycoords='data',        # xy: goal
                        xytext=posArr_Pred[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color='b'))

        ax.scatter(posArr_Pred[0][0], posArr_Pred[0][1], s = 150, c = 'red',alpha = 1, marker='s')
        ax.scatter(posArr_Pred[-1][0], posArr_Pred[-1][1], s = 150, c = 'red',alpha = 1, marker='o')      # plot the ending point
        minPosX_pred = np.amin(posArr_Pred[:,0]); maxPosX_pred = np.amax(posArr_Pred[:,0]);  # calculate min and max value of x-axis
        minPosY_pred = np.amin(posArr_Pred[:,1]); maxPosY_pred = np.amax(posArr_Pred[:,1]);  # calculate min and max value of y-axis
        minX = minPosX_org if minPosX_org <= minPosX_pred  else minPosX_pred
        maxX = maxPosX_org if maxPosX_org >  maxPosX_pred  else maxPosX_pred
        minY = minPosY_org if minPosY_org <= minPosY_pred  else minPosY_pred
        maxY = maxPosY_org if maxPosY_org >  maxPosY_pred  else maxPosY_pred
        plt.xlim(minX*0.5, maxX*1.5)    # set the limit of x-axis
        plt.ylim(minY*0.5, maxY*1.5)    # set the limit of y-axis
    
    else:
        plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
        plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
        
    ax.set_ylim(ax.get_ylim()[::-1])      # invert the y-axis
    plt.show()
    
    
    
    

def plotTrack_mult(dataMat_org, dataMat_Pred=None, predShow =False, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data
    plt.figure(1, figsize=fig_size)
#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")             # set begiing and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis
    maxStep_x = (maxPosX_org - minPosX_org)*0.8;  maxStep_y = (maxPosY_org - minPosY_org)*0.8
    
    for i in range(np.shape(posArr_org)[0]-1):     # plot the original route
        if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  maxStep_x) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  maxStep_y)): continue
        ax.annotate("",
                    xy=posArr_org[i+1], xycoords='data',        # xy: goal
                    xytext=posArr_org[i], textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3"))
                    
#   Since we have more than one route, we can't determin the unique start and end point
#    ax.scatter(posArr_org[0][0], posArr_org[0][1], s = 100, c = 'green',alpha = 0.8, marker='s')       # plot the beginning point
#    ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 100, c = 'green',alpha = 1, marker='o')      # plot the ending point
    
    if (predShow == True):                   # plot the predicted route     
        posArr_Pred = getPosArr(dataMat_Pred)
        for i in range(np.shape(posArr_Pred)[0]-1):     
            ax.annotate("",
                        xy=posArr_Pred[i+1], xycoords='data',        # xy: goal
                        xytext=posArr_Pred[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color='red'))
             
        ax.scatter(posArr_Pred[0][0], posArr_Pred[0][1], s = 150, c = 'red',alpha = 1, marker='s')
        ax.scatter(posArr_Pred[-1][0], posArr_Pred[-1][1], s = 150, c = 'red',alpha = 1, marker='o')      # plot the ending point
        
        if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
            plt.xlim(minX_plot, maxX_plot)    # set the limit of x-axis
            plt.ylim(minY_plot, maxY_plot)    # set the limit of y-axis
        else:
            minPosX_pred = np.amin(posArr_Pred[:,0]); maxPosX_pred = np.amax(posArr_Pred[:,0]);  # calculate min and max value of x-axis
            minPosY_pred = np.amin(posArr_Pred[:,1]); maxPosY_pred = np.amax(posArr_Pred[:,1]);  # calculate min and max value of y-axis
            minX = minPosX_org if minPosX_org <= minPosX_pred  else minPosX_pred
            maxX = maxPosX_org if maxPosX_org >  maxPosX_pred  else maxPosX_pred
            minY = minPosY_org if minPosY_org <= minPosY_pred  else minPosY_pred
            maxY = maxPosY_org if maxPosY_org >  maxPosY_pred  else maxPosY_pred
            plt.xlim(minX*0.5, maxX*1.5)    # set the limit of x-axis
            plt.ylim(minY*0.5, maxY*1.5)    # set the limit of y-axis
    
    else:
        if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
            plt.xlim(minX_plot, maxX_plot)    # set the limit of x-axis
            plt.ylim(minY_plot, maxY_plot)    # set the limit of y-axis
        else:
            plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
            plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
        
    ax.set_ylim(ax.get_ylim()[::-1])      # invert the y-axis
    plt.show()
    

def plot_1234(dataMat_org, dataMat_Pred=None, origShow=False, predShow =True, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None, model = "regTree"):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data
    plt.figure(1, figsize=fig_size)
#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")             # set begiing and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis
    maxStep_x = (maxPosX_org - minPosX_org)*0.8;  maxStep_y = (maxPosY_org - minPosY_org)*0.8
    
    ### plot the original route
    if (origShow == True): 
        for i in range(np.shape(posArr_org)[0]-1):     
            if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  maxStep_x) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  maxStep_y)): continue  ## if it is the ending point of one trace, then skip the connection this time!
            ax.annotate("",
                        xy=posArr_org[i+1], xycoords='data',        # xy: goal
                        xytext=posArr_org[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3"))
                    
#   Since we have more than one route, we can't determin the unique start and end point
#    ax.scatter(posArr_org[0][0], posArr_org[0][1], s = 100, c = 'green',alpha = 0.8, marker='s')       # plot the beginning point
#    ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 100, c = 'green',alpha = 1, marker='o')      # plot the ending point
    
    ### plot the predicted route 
    if (predShow == True):                  
        if (model == "regTree"):        arrow_color = 'red'
        elif (model == "modelTree"):    arrow_color = 'blue'
        elif (model == "regNN"):        arrow_color = 'green'
        else:                           arrow_color = 'black'
        
        posArr_Pred = getPosArr(dataMat_Pred)
        print(np.shape(posArr_Pred))               
        for i in range(np.shape(posArr_Pred)[0]-1):     
            ax.annotate("",
                        xy=posArr_Pred[i+1], xycoords='data',        # xy: goal
                        xytext=posArr_Pred[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = arrow_color))
        
        
        #history_step_num = (dataMat_Pred.shape[1]-2)/4       
        #start_steps = np.array(dataMat_regNN_Pred[0,:]).ravel()
        
        #for i in range(history_step_num): 
        #ax.scatter(start_steps[0*i], start_steps[0*i+1], s = 150, c = 'black',alpha = 1, marker='s')
        ax.scatter(posArr_Pred[0][0], posArr_Pred[0][1], s = 150, c = 'red',alpha = 1, marker='s')
        ax.scatter(posArr_Pred[-1][0], posArr_Pred[-1][1], s = 150, c = 'red',alpha = 1, marker='o')      # plot the ending point       
        
        if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
            plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
            plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
        else:
            minPosX_pred = np.amin(posArr_Pred[:,0]); maxPosX_pred = np.amax(posArr_Pred[:,0]);  # calculate min and max value of x-axis
            minPosY_pred = np.amin(posArr_Pred[:,1]); maxPosY_pred = np.amax(posArr_Pred[:,1]);  # calculate min and max value of y-axis
            minX = minPosX_org if minPosX_org <= minPosX_pred  else minPosX_pred
            maxX = maxPosX_org if maxPosX_org >  maxPosX_pred  else maxPosX_pred
            minY = minPosY_org if minPosY_org <= minPosY_pred  else minPosY_pred
            maxY = maxPosY_org if maxPosY_org >  maxPosY_pred  else maxPosY_pred
            plt.xlim(minX*0.5, maxX*1.5)        # set the limit of x-axis
            plt.ylim(minY*0.5, maxY*1.5)        # set the limit of y-axis
    
    else:
        if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
            plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
            plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
        else:
            plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
            plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
        
    ax.set_ylim(ax.get_ylim()[::-1])            # invert the y-axis
    
    ### plot the blocks!
    plt.plot([0,210],[265,265], 'k'); plt.plot([0,210],[120,120], 'k'); plt.plot([210,210],[120,265], 'k')
    plt.plot([430,640],[120,120], 'k'); plt.plot([430,640],[265,265], 'k'); plt.plot([430,430],[120,265], 'k')
    plt.plot([284,350],[243,243], 'red'); plt.plot([350,350],[243,149], 'red'); plt.plot([350,284],[149,149], 'red'); plt.plot([284,284],[149,243], 'red')
    
    ### plot gates 1,2,3,4
    plt.plot([0,20],[3,3], 'red'); plt.plot([0,20],[110,110], 'red');  plt.plot([20,20],[3,110], 'red') 
    plt.plot([620,640],[3,3], 'blue'); plt.plot([620,640],[110,110], 'blue');  plt.plot([620,620],[3,110], 'blue') 
    plt.plot([0,20],[275,275], 'cyan'); plt.plot([0,20],[390,390], 'cyan');  plt.plot([20,20],[275,390], 'cyan') 
    plt.plot([620,640],[275,275], 'green'); plt.plot([620,640],[390,390], 'green');  plt.plot([620,620],[275,390], 'green') 

    plt.show()
    
    
def plot_1234_v2(dataMat_org, dataMat_Pred=None, origShow=False, predShow =True, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None, model = "regTree"):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data
    plt.figure(1, figsize=fig_size)
#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")             # set begiing and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis
    maxStep_x = (maxPosX_org - minPosX_org)*0.8;  maxStep_y = (maxPosY_org - minPosY_org)*0.8
    
    ### plot the original route
    if (origShow == True): 
        for i in range(np.shape(posArr_org)[0]-1):     
            if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  maxStep_x) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  maxStep_y)): continue  ## if it is the ending point of one trace, then skip the connection this time!
            ax.annotate("",
                        xy=posArr_org[i+1], xycoords='data',        # xy: goal
                        xytext=posArr_org[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3"))
                    
#   Since we have more than one route, we can't determin the unique start and end point
#    ax.scatter(posArr_org[0][0], posArr_org[0][1], s = 100, c = 'green',alpha = 0.8, marker='s')       # plot the beginning point
#    ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 100, c = 'green',alpha = 1, marker='o')      # plot the ending point
    
    
    
    ### plot the predicted route 

    if (predShow == True):  
    
        ## convert the pos_matrix to array
        history_step_num = (dataMat_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_Pred[1:,:])
        
        ## define the arrow color
        if (model == "regTree"):        arrow_color = 'magenta'
        elif (model == "modTree"):      arrow_color = 'cyan'
        elif (model == "regNN"):        arrow_color = 'green'
        else:                           arrow_color = 'black'
        
        ## connect all history steps

        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = arrow_color))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = arrow_color))
        
        
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

      
        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = arrow_color,alpha = 1, marker='o')      
        
        if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
            plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
            plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
        else:
            minPosX_pred = np.amin(posArr_Pred[:,0]); maxPosX_pred = np.amax(posArr_Pred[:,0]);  # calculate min and max value of x-axis
            minPosY_pred = np.amin(posArr_Pred[:,1]); maxPosY_pred = np.amax(posArr_Pred[:,1]);  # calculate min and max value of y-axis
            minX = minPosX_org if minPosX_org <= minPosX_pred  else minPosX_pred
            maxX = maxPosX_org if maxPosX_org >  maxPosX_pred  else maxPosX_pred
            minY = minPosY_org if minPosY_org <= minPosY_pred  else minPosY_pred
            maxY = maxPosY_org if maxPosY_org >  maxPosY_pred  else maxPosY_pred
            plt.xlim(minX*0.5, maxX*1.5)        # set the limit of x-axis
            plt.ylim(minY*0.5, maxY*1.5)        # set the limit of y-axis
    
    else:
        if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
            plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
            plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
        else:
            plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
            plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
        
    ax.set_ylim(ax.get_ylim()[::-1])            # invert the y-axis
    
    ### plot the blocks!
    plt.plot([0,210],[265,265], 'k'); plt.plot([0,210],[120,120], 'k'); plt.plot([210,210],[120,265], 'k')
    plt.plot([430,640],[120,120], 'k'); plt.plot([430,640],[265,265], 'k'); plt.plot([430,430],[120,265], 'k')
    plt.plot([284,350],[243,243], 'red'); plt.plot([350,350],[243,149], 'red'); plt.plot([350,284],[149,149], 'red'); plt.plot([284,284],[149,243], 'red')
    
    ### plot gates 1,2,3,4
    plt.plot([0,20],[3,3], 'red'); plt.plot([0,20],[110,110], 'red');  plt.plot([20,20],[3,110], 'red') 
    plt.plot([620,640],[3,3], 'blue'); plt.plot([620,640],[110,110], 'blue');  plt.plot([620,620],[3,110], 'blue') 
    plt.plot([0,20],[275,275], 'cyan'); plt.plot([0,20],[390,390], 'cyan');  plt.plot([20,20],[275,390], 'cyan') 
    plt.plot([620,640],[275,275], 'green'); plt.plot([620,640],[390,390], 'green');  plt.plot([620,620],[275,390], 'green') 
    

    plt.show()
    
    
    
####################################
####################################
####################################


def plot_1234_v3(dataMat_org, dataMat_regTree_Pred=None, dataMat_modTree_Pred=None, dataMat_regNN_Pred=None, dataMat_error=None, origShow=False, predShow =False, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None, model = "regTree"):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data
    plt.figure(1, figsize=fig_size)
    plt.cla()   ## clean the drawing
#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")             # set begiing and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis
    maxStep_x = (maxPosX_org - minPosX_org)*0.8;  maxStep_y = (maxPosY_org - minPosY_org)*0.8
    
    ###################
    ### plot the original route
    if (origShow == True): 
        for i in range(np.shape(posArr_org)[0]-1):     
            #if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  maxStep_x) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  maxStep_y)): continue  ## if it is the ending point of one trace, then skip the connection this time!
            if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  40) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  40)):
                ax.annotate("",
            #                xy=(posArr_org[i][0]+dataMat_org[i,2]+dataMat_org[i,4], posArr_org[i][1]+dataMat_org[i,3]+dataMat_org[i,5]), xycoords='data', 
                            xy=(posArr_org[i][0]+dataMat_org[i,2], posArr_org[i][1]+dataMat_org[i,3]), xycoords='data',            # xy: goal
                            xytext=posArr_org[i], textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3"))
            
            
            else:               
                ax.annotate("",
                            xy=posArr_org[i+1], xycoords='data',        # xy: goal
                            xytext=posArr_org[i], textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3"))
                    

                    
#   Since we have more than one route, we can't determin the unique start and end point
#    ax.scatter(posArr_org[0][0], posArr_org[0][1], s = 100, c = 'green',alpha = 0.8, marker='s')       # plot the beginning point
#    ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 100, c = 'green',alpha = 1, marker='o')      # plot the ending point
    
    ###################
    ## plot error steps 
    if (np.all(dataMat_error) != None and predShow == True):        ## dataMat_error: [num of errors, num of steps of each single error]
        ## convert the pos_matrix to array
        history_step_num = dataMat_error.shape[1] // 4
        
        for line in range(0, dataMat_error.shape[0]-1):             ## populate different errors
            hist_pos_arr = np.array(dataMat_error[line,:]).ravel()
            
            ## connect all history steps
            for i in range(history_step_num-1):                     ## show all steps within one error
                ax.annotate("",
                            xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                            xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3", color = 'black'))
            
            ## show the last hist. step = last position + last speed
            ax.annotate("",
                        xy=(hist_pos_arr[-4] + hist_pos_arr[-2], hist_pos_arr[-3] + hist_pos_arr[-1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[-4], hist_pos_arr[-3]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
            
        
    ##########################################
    ## plot steps predicted by regression tree 
    if (np.all(dataMat_regTree_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_regTree_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_regTree_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_regTree_Pred[1:,:])

        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'magenta'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'magenta'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'magenta', alpha = 1, marker='o')      
        
        
    #####################################
    ## plot steps predicted by model tree   
    if (np.all(dataMat_modTree_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_modTree_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_modTree_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_modTree_Pred[1:,:])

        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'cyan'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'cyan'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'cyan', alpha = 1, marker='o')      
        
        
    #########################################
    ## plot steps predicted by neural network 
    if (np.all(dataMat_regNN_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_regNN_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_regNN_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_regNN_Pred[1:,:])
        
        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'green'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'green'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'green', alpha = 1, marker='o') 
        

    # if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
        # plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
        # plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
    # else:
            # minPosX_pred = np.amin(posArr_Pred[:,0]); maxPosX_pred = np.amax(posArr_Pred[:,0]);  # calculate min and max value of x-axis
            # minPosY_pred = np.amin(posArr_Pred[:,1]); maxPosY_pred = np.amax(posArr_Pred[:,1]);  # calculate min and max value of y-axis
            # minX = minPosX_org if minPosX_org <= minPosX_pred  else minPosX_pred
            # maxX = maxPosX_org if maxPosX_org >  maxPosX_pred  else maxPosX_pred
            # minY = minPosY_org if minPosY_org <= minPosY_pred  else minPosY_pred
            # maxY = maxPosY_org if maxPosY_org >  maxPosY_pred  else maxPosY_pred
            # plt.xlim(minX*0.5, maxX*1.5)        # set the limit of x-axis
            # plt.ylim(minY*0.5, maxY*1.5)        # set the limit of y-axis
    

    if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
        plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
        plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
    else:
        plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
        plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
    
    # invert the y-axis    
    ax.set_ylim(ax.get_ylim()[::-1])            


    ### plot the blocks!
    plt.plot([0,210],[265,265], 'k'); plt.plot([0,210],[120,120], 'k'); plt.plot([210,210],[120,265], 'k')
    plt.plot([430,640],[120,120], 'k'); plt.plot([430,640],[265,265], 'k'); plt.plot([430,430],[120,265], 'k')
    plt.plot([284,350],[243,243], 'red'); plt.plot([350,350],[243,149], 'red'); plt.plot([350,284],[149,149], 'red'); plt.plot([284,284],[149,243], 'red')
    
    ### plot gates 1,2,3,4
    plt.plot([0,20],[3,3], 'red'); plt.plot([0,20],[110,110], 'red');  plt.plot([20,20],[3,110], 'red') 
    plt.plot([620,640],[3,3], 'blue'); plt.plot([620,640],[110,110], 'blue');  plt.plot([620,620],[3,110], 'blue') 
    plt.plot([0,20],[275,275], 'cyan'); plt.plot([0,20],[390,390], 'cyan');  plt.plot([20,20],[275,390], 'cyan') 
    plt.plot([620,640],[275,275], 'green'); plt.plot([620,640],[390,390], 'green');  plt.plot([620,620],[275,390], 'green') 

    plt.show()
    

def plot_trajectory(dataMat_org, dataMat_regTree_Pred=None, dataMat_modTree_Pred=None, dataMat_regNN_Pred=None, dataMat_error=None, origShow=False, predShow =False, plotEachStep=False, fig_size = (10,10), minStepSize=4, maxStepSize=7, minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None, map = "1234"):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data

    
    plt.figure(1, figsize=fig_size)
    plt.cla()                                                 ## clean the drawing

    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")             # set beginning and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis
    maxStep_x = (maxPosX_org - minPosX_org)*0.8;  maxStep_y = (maxPosY_org - minPosY_org)*0.8
    
    ###################
    ### plot the original route
    if (origShow == True): 
#        print(posArr_org)
        if map == "1234":
            for i in range(np.shape(posArr_org)[0]-1):     
                # if next point is from another trajectory, then finish the current trajectory
                if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  maxStep_x) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  maxStep_y)):   ## if it is the ending point of one trace, then skip the connection this time!
                    ax.annotate("",
                                xy=(posArr_org[i][0]+dataMat_org[i,2], posArr_org[i][1]+dataMat_org[i,3]), xycoords='data',        # xy: goal
                                xytext=posArr_org[i], textcoords='data',    # xytest: parent
                                bbox=posBegEnd,
                                arrowprops=dict(arrowstyle='-|>',
                                connectionstyle="arc3"))
    #                print("\nmark-1")            
                ## if next point is from the same trajectory, then connect these points
                else:
                    ax.annotate("",
                                xy=posArr_org[i+1], xycoords='data',        # xy: goal
                                xytext=posArr_org[i], textcoords='data',    # xytest: parent
                                bbox=posBegEnd,
                                arrowprops=dict(arrowstyle='-|>',
                                connectionstyle="arc3"))
    #                print("\nmark-2")  
        
        elif map == "forum":
            for i in range(np.shape(posArr_org)[0]-1):     
                # if next point is from another trajectory, then finish the current trajectory
                if np.sqrt(np.square(posArr_org[i+1][0] - posArr_org[i][0]) + np.square(posArr_org[i+1][1] - posArr_org[i][1])) >= maxStepSize:   ## if the distance is larger than 10 pixels
                    ax.annotate("",
                                xy=(posArr_org[i][0]+dataMat_org[i,2], posArr_org[i][1]+dataMat_org[i,3]), xycoords='data',        # xy: goal
                                xytext=posArr_org[i], textcoords='data',    # xytest: parent
                                bbox=posBegEnd,
                                arrowprops=dict(arrowstyle='-|>',
                                connectionstyle="arc3"))
    #                print("\nmark-3")            
                ## if next point is from the same trajectory, then connect these points
                else:
                    ax.annotate("",
                                xy=posArr_org[i+1], xycoords='data',        # xy: goal
                                xytext=posArr_org[i], textcoords='data',    # xytest: parent
                                bbox=posBegEnd,
                                arrowprops=dict(arrowstyle='-|>',
                                connectionstyle="arc3"))
    #                print("\nmark-4")  
                
        ## show the last step = last position + last speed
        ax.annotate("",
                    xy=(posArr_org[-1][0]+dataMat_org[-1,2], posArr_org[-1][1]+dataMat_org[-1,3]), xycoords='data',        # xy: goal
                    xytext=(posArr_org[-1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'black'))
                    
        ## plot each steps 
        if plotEachStep == True:
            for i in range(len(posArr_org)): 
                ax.scatter(posArr_org[i][0], posArr_org[i][1], s = 50, c = 'black',alpha = 1, marker='s')
            
        ## plot last step          
        #ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 50, c = 'black', alpha = 1, marker='s') 
                    

    ###################
    ## plot error steps 
    if (np.all(dataMat_error) != None and predShow == True):        ## dataMat_error: [num of errors, num of steps of each single error]
        ## convert the pos_matrix to array
        history_step_num = dataMat_error.shape[1] // 4
        
        for line in range(0, dataMat_error.shape[0]-1):             ## populate different errors
            hist_pos_arr = np.array(dataMat_error[line,:]).ravel()
            
            ## connect all history steps
            for i in range(history_step_num-1):                     ## show all steps within one error
                ax.annotate("",
                            xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                            xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3", color = 'black'))
            
            ## show the last hist. step = last position + last speed
            ax.annotate("",
                        xy=(hist_pos_arr[-4] + hist_pos_arr[-2], hist_pos_arr[-3] + hist_pos_arr[-1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[-4], hist_pos_arr[-3]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
            

    ##########################################
    ## plot steps predicted by regression tree 
    if (np.all(dataMat_regTree_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_regTree_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_regTree_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_regTree_Pred[1:,:])

        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'magenta'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'magenta'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'magenta', alpha = 1, marker='o')      
        
        
    #####################################
    ## plot steps predicted by model tree   
    if (np.all(dataMat_modTree_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_modTree_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_modTree_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_modTree_Pred[1:,:])

        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'cyan'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'cyan'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'cyan', alpha = 1, marker='o')      
        
        
    #########################################
    ## plot steps predicted by neural network 
    if (np.all(dataMat_regNN_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_regNN_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_regNN_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_regNN_Pred[1:,:])
        
        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'green'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'green'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'green', alpha = 1, marker='o') 
        

    
    ## setting x, y-axis
    if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
        plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
        plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
        
    else:
        if map == 'forum':
        
            plt.xlim(0, 640)      # set the limit of x-axis
            plt.ylim(0, 480)      # set the limit of y-axis
        
            ### plot Auditorium
            plt.plot([0,60],[300,300], 'k'); plt.plot([60,60],[0,300], 'k');
            
            ### plot Cafe
            plt.plot([215,215],[0,60], 'k'); plt.plot([60,60],[0,60], 'k'); plt.plot([215,60],[60,60], 'k')
            
            ### plot Stairs
            plt.plot([215,330],[60,60], 'k'); plt.plot([215,215],[0,60], 'k'); plt.plot([330,330],[0,60], 'k')

            ### plot Elevator
            plt.plot([445,570],[60,60], 'k'); plt.plot([445,445],[0,60], 'k'); plt.plot([570,570],[0,60], 'k')
            
            ### plot Night exit door
            plt.plot([570,570],[0,60], 'k'); plt.plot([570,640],[60,60], 'k'); 
            
            ### plot Robot lab
            plt.plot([570,640],[60,60], 'k'); plt.plot([570,640],[420,420], 'k'); plt.plot([570,570],[60,420], 'k')
            
            ### plot Vision lab
            plt.plot([550,640],[420,420], 'k'); plt.plot([550,550],[420,480], 'k'); 
            
            ### plot Reception
            plt.plot([200,300],[420,420], 'k'); plt.plot([200,200],[420,480], 'k'); plt.plot([300,300],[420,480], 'k')
            
            ### plot Main entry
            plt.plot([0,180],[420,420], 'k'); plt.plot([180,180],[420,480], 'k'); 
        
        
        elif map == '1234':
        
            plt.xlim(0, 640)      # set the limit of x-axis
            plt.ylim(0, 400)      # set the limit of y-axis
        
            ### plot the blocks!
            plt.plot([0,210],[265,265], 'k'); plt.plot([0,210],[120,120], 'k'); plt.plot([210,210],[120,265], 'k')
            plt.plot([430,640],[120,120], 'k'); plt.plot([430,640],[265,265], 'k'); plt.plot([430,430],[120,265], 'k')
            plt.plot([284,350],[243,243], 'red'); plt.plot([350,350],[243,149], 'red'); plt.plot([350,284],[149,149], 'red'); plt.plot([284,284],[149,243], 'red')
            
            ### plot gates 1,2,3,4
            plt.plot([0,20],[3,3], 'red'); plt.plot([0,20],[110,110], 'red');  plt.plot([20,20],[3,110], 'red') 
            plt.plot([620,640],[3,3], 'blue'); plt.plot([620,640],[110,110], 'blue');  plt.plot([620,620],[3,110], 'blue') 
            plt.plot([0,20],[275,275], 'cyan'); plt.plot([0,20],[390,390], 'cyan');  plt.plot([20,20],[275,390], 'cyan') 
            plt.plot([620,640],[275,275], 'green'); plt.plot([620,640],[390,390], 'green');  plt.plot([620,620],[275,390], 'green') 

        else:
            plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
            plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
    
    
    # invert the y-axis    
    ax.set_ylim(ax.get_ylim()[::-1])            

    plt.show()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def plot_forum(dataMat_org, dataMat_regTree_Pred=None, dataMat_modTree_Pred=None, dataMat_regNN_Pred=None, dataMat_error=None, origShow=False, predShow =False, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None, model = "regTree"):
    posArr_org = getPosArr(dataMat_org)                       # extract the position data
    plt.figure(1, figsize=fig_size)
    plt.cla()   ## clean the drawing
#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    posBegEnd = dict(boxstyle="round4", fc="0.8")             # set begiing and ending format

    minPosX_org = np.amin(posArr_org[:,0]); maxPosX_org = np.amax(posArr_org[:,0]);  # calculate min and max value of x-axis
    minPosY_org = np.amin(posArr_org[:,1]); maxPosY_org = np.amax(posArr_org[:,1]);  # calculate min and max value of y-axis
    maxStep_x = (maxPosX_org - minPosX_org)*0.8;  maxStep_y = (maxPosY_org - minPosY_org)*0.8
    
    ###################
    ### plot the original route
    if (origShow == True): 
        for i in range(np.shape(posArr_org)[0]-1):     
            #if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  maxStep_x) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  maxStep_y)): continue  ## if it is the ending point of one trace, then skip the connection this time!
            #if((np.abs(posArr_org[i+1][0] - posArr_org[i][0]) >  10) or (np.abs(posArr_org[i+1][1] - posArr_org[i][1]) >  10)): continue  ## if the distance between two steps is too big, then don't connect them!
            
            # if next point is from another trajectory, then finish the current trajectory
            if np.sqrt(np.square(posArr_org[i+1][0] - posArr_org[i][0]) + np.square(posArr_org[i+1][1] - posArr_org[i][1])) >= 10: 
                ax.annotate("",
                            xy=(posArr_org[i][0]+dataMat_org[i,2]+dataMat_org[i,4], posArr_org[i][1]+dataMat_org[i,3]+dataMat_org[i,5]), xycoords='data',        # xy: goal
                            xytext=posArr_org[i], textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3"))
#                print("\nmark-1")            
            # if next point is from the same trajectory, then connect these points
            else:
                ax.annotate("",
                            xy=posArr_org[i+1], xycoords='data',        # xy: goal
                            xytext=posArr_org[i], textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3"))
#                print("\nmark-2")  
                
        ## show the last step = last position + last speed
        ax.annotate("",
                    xy=(posArr_org[-1][0]+dataMat_org[-1,2]+dataMat_org[-1,4], posArr_org[-1][1]+dataMat_org[-1,3]+dataMat_org[-1,5]), xycoords='data',        # xy: goal
                    xytext=(posArr_org[-1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'black'))
#        print("\nmark-3")  
#   Since we have more than one route, we can't determin the unique start and end point
#    ax.scatter(posArr_org[0][0], posArr_org[0][1], s = 100, c = 'green',alpha = 0.8, marker='s')       # plot the beginning point
#    ax.scatter(posArr_org[-1][0], posArr_org[-1][1], s = 100, c = 'green',alpha = 1, marker='o')      # plot the ending point
    
    ###################
    ## plot error steps 
    if (np.all(dataMat_error) != None and predShow == True):        ## dataMat_error: [num of errors, num of steps of each single error]
        ## convert the pos_matrix to array
        history_step_num = dataMat_error.shape[1] // 4
        
        for line in range(0, dataMat_error.shape[0]-1):             ## populate different errors
            hist_pos_arr = np.array(dataMat_error[line,:]).ravel()
            
            ## connect all history steps
            for i in range(history_step_num-1):                     ## show all steps within one error
                ax.annotate("",
                            xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                            xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                            bbox=posBegEnd,
                            arrowprops=dict(arrowstyle='-|>',
                            connectionstyle="arc3", color = 'black'))
            
            ## show the last hist. step = last position + last speed
            ax.annotate("",
                        xy=(hist_pos_arr[-4] + hist_pos_arr[-2], hist_pos_arr[-3] + hist_pos_arr[-1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[-4], hist_pos_arr[-3]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
            
        
    ##########################################
    ## plot steps predicted by regression tree 
    if (np.all(dataMat_regTree_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_regTree_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_regTree_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_regTree_Pred[1:,:])

        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'magenta'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'magenta'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'magenta', alpha = 1, marker='o')      
        
        
    #####################################
    ## plot steps predicted by model tree   
    if (np.all(dataMat_modTree_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_modTree_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_modTree_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_modTree_Pred[1:,:])

        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'cyan'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'cyan'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'cyan', alpha = 1, marker='o')      
        
        
    #########################################
    ## plot steps predicted by neural network 
    if (np.all(dataMat_regNN_Pred) != None and predShow == True):  
        ## convert the pos_matrix to array
        history_step_num = (dataMat_regNN_Pred.shape[1]-2) // 4
        hist_pos_arr = np.array(dataMat_regNN_Pred[0,:]).ravel()
        prediction_pos_arr = getPosArr(dataMat_regNN_Pred[1:,:])
        
        ## connect all history steps
        for i in range(history_step_num-1):     
            ax.annotate("",
                        xy=(hist_pos_arr[(i+1)*4], hist_pos_arr[(i+1)*4+1]), xycoords='data',        # xy: goal
                        xytext=(hist_pos_arr[i*4], hist_pos_arr[i*4+1]), textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'black'))
        
        # connect the last history step and the first predicted point
        ax.annotate("",
                    xy=prediction_pos_arr[0], xycoords='data',        # xy: goal
                    xytext=(hist_pos_arr[(history_step_num-1)*4], hist_pos_arr[(history_step_num-1)*4+1]), textcoords='data',    # xytest: parent
                    bbox=posBegEnd,
                    arrowprops=dict(arrowstyle='-|>',
                    connectionstyle="arc3", color = 'green'))
        
        ## plot all predicted steps
        for i in range(np.shape(prediction_pos_arr)[0]-1):     
            ax.annotate("",
                        xy=prediction_pos_arr[i+1], xycoords='data',        # xy: goal
                        xytext=prediction_pos_arr[i], textcoords='data',    # xytest: parent
                        bbox=posBegEnd,
                        arrowprops=dict(arrowstyle='-|>',
                        connectionstyle="arc3", color = 'green'))
        
        # plot the starting point
        for i in range(history_step_num): 
            ax.scatter(hist_pos_arr[i*4], hist_pos_arr[i*4+1], s = 50, c = 'black',alpha = 1, marker='s')

        # plot the ending point      
        ax.scatter(prediction_pos_arr[-1][0], prediction_pos_arr[-1][1], s = 150, c = 'green', alpha = 1, marker='o') 
        

    # if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
        # plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
        # plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
    # else:
            # minPosX_pred = np.amin(posArr_Pred[:,0]); maxPosX_pred = np.amax(posArr_Pred[:,0]);  # calculate min and max value of x-axis
            # minPosY_pred = np.amin(posArr_Pred[:,1]); maxPosY_pred = np.amax(posArr_Pred[:,1]);  # calculate min and max value of y-axis
            # minX = minPosX_org if minPosX_org <= minPosX_pred  else minPosX_pred
            # maxX = maxPosX_org if maxPosX_org >  maxPosX_pred  else maxPosX_pred
            # minY = minPosY_org if minPosY_org <= minPosY_pred  else minPosY_pred
            # maxY = maxPosY_org if maxPosY_org >  maxPosY_pred  else maxPosY_pred
            # plt.xlim(minX*0.5, maxX*1.5)        # set the limit of x-axis
            # plt.ylim(minY*0.5, maxY*1.5)        # set the limit of y-axis
    

    if (minX_plot != None and maxX_plot != None and minY_plot != None and maxY_plot != None):
        plt.xlim(minX_plot, maxX_plot)      # set the limit of x-axis
        plt.ylim(minY_plot, maxY_plot)      # set the limit of y-axis
    else:
        plt.xlim(minPosX_org*0.5, maxPosX_org*1.5)    # set the limit of x-axis
        plt.ylim(minPosY_org*0.5, maxPosY_org*1.5)    # set the limit of y-axis
    
    # invert the y-axis    
    ax.set_ylim(ax.get_ylim()[::-1])            

    
    ### plot Auditorium
    plt.plot([0,60],[300,300], 'k'); plt.plot([60,60],[0,300], 'k');  
    
    ### plot Cafe
    plt.plot([215,215],[0,60], 'k'); plt.plot([60,60],[0,60], 'k'); plt.plot([215,60],[60,60], 'k') 
    
    ### plot Stairs
    plt.plot([215,330],[60,60], 'k'); plt.plot([215,215],[0,60], 'k'); plt.plot([330,330],[0,60], 'k')

    ### plot Elevator
    plt.plot([445,570],[60,60], 'k'); plt.plot([445,445],[0,60], 'k'); plt.plot([570,570],[0,60], 'k')
    
    ### plot Night exit door
    plt.plot([570,570],[0,60], 'k'); plt.plot([570,640],[60,60], 'k'); 
    
    ### plot Robot lab
    plt.plot([570,640],[60,60], 'k'); plt.plot([570,640],[420,420], 'k'); plt.plot([570,570],[60,420], 'k')
    
    ### plot Vision lab
    plt.plot([550,640],[420,420], 'k'); plt.plot([550,550],[420,480], 'k'); 
    
    ### plot Reception
    plt.plot([200,300],[420,420], 'k'); plt.plot([200,200],[420,480], 'k'); plt.plot([300,300],[420,480], 'k')
    
    ### plot Main entry
    plt.plot([0,180],[420,420], 'k'); plt.plot([180,180],[420,480], 'k'); 
    
    '''   
    1:  main entry:    [0,420], [180,420], [180, 480], [0, 480]
    2:  Auditorium, [0,0], [60,0], [60,300], [0,300]
    3:  Cafe, [60,0], [215,0], [215,60], [60,60]
    4:  Stairs, [215,0], [330,0], [330,60], [215,60]
    5:  Elevator, [445,0], [570,0], [570,60], [445,60]
    6:  night exit door: [570,0], [640,0], [640,60], [570,60]
    7:  robot lab:      [570,60], [640,60], [640, 420], [570, 420]
    8:  vision lab:     [550,420], [640,420], [640, 480], [550, 480]
    9:  reception:     [200,420], [300,420], [300, 480], [200, 480]

    ''' 
    plt.show()
    
    
def plot_position(destinationMat_set, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None, map = "1234"):
    posArr_org = getPosArr(destinationMat_set)                       # extract the position data
    plt.figure(1, figsize=fig_size)
    plt.cla()   ## clean the drawing

#    axprops = dict(xticks=[], yticks=[])
#    ax = plt.subplot(111, frameon=False, **axprops)          # delete x- and y-axis
    ax = plt.subplot(111)                                     # keep x- and y-axis
    for i in range(np.shape(posArr_org)[0]):     # plot the original route
        ax.scatter(posArr_org[i][0], posArr_org[i][1], s = 50, c = 'black',alpha = 1, marker='s')       # plot the beginning point
        print(posArr_org)
        
    if map == 'forum':
    
        plt.xlim(0, 640)      # set the limit of x-axis
        plt.ylim(0, 480)      # set the limit of y-axis
    
        ### plot Auditorium
        plt.plot([0,60],[300,300], 'k'); plt.plot([60,60],[0,300], 'k');
        
        ### plot Cafe
        plt.plot([215,215],[0,60], 'k'); plt.plot([60,60],[0,60], 'k'); plt.plot([215,60],[60,60], 'k')
        
        ### plot Stairs
        plt.plot([215,330],[60,60], 'k'); plt.plot([215,215],[0,60], 'k'); plt.plot([330,330],[0,60], 'k')

        ### plot Elevator
        plt.plot([445,570],[60,60], 'k'); plt.plot([445,445],[0,60], 'k'); plt.plot([570,570],[0,60], 'k')
        
        ### plot Night exit door
        plt.plot([570,570],[0,60], 'k'); plt.plot([570,640],[60,60], 'k'); 
        
        ### plot Robot lab
        plt.plot([570,640],[60,60], 'k'); plt.plot([570,640],[420,420], 'k'); plt.plot([570,570],[60,420], 'k')
        
        ### plot Vision lab
        plt.plot([550,640],[420,420], 'k'); plt.plot([550,550],[420,480], 'k'); 
        
        ### plot Reception
        plt.plot([200,300],[420,420], 'k'); plt.plot([200,200],[420,480], 'k'); plt.plot([300,300],[420,480], 'k')
        
        ### plot Main entry
        plt.plot([0,180],[420,420], 'k'); plt.plot([180,180],[420,480], 'k'); 
    
    
    elif map == '1234':
    
        plt.xlim(0, 640)      # set the limit of x-axis
        plt.ylim(0, 400)      # set the limit of y-axis
    
        ### plot the blocks!
        plt.plot([0,210],[265,265], 'k'); plt.plot([0,210],[120,120], 'k'); plt.plot([210,210],[120,265], 'k')
        plt.plot([430,640],[120,120], 'k'); plt.plot([430,640],[265,265], 'k'); plt.plot([430,430],[120,265], 'k')
        plt.plot([284,350],[243,243], 'red'); plt.plot([350,350],[243,149], 'red'); plt.plot([350,284],[149,149], 'red'); plt.plot([284,284],[149,243], 'red')
        
        ### plot gates 1,2,3,4
        plt.plot([0,20],[3,3], 'red'); plt.plot([0,20],[110,110], 'red');  plt.plot([20,20],[3,110], 'red') 
        plt.plot([620,640],[3,3], 'blue'); plt.plot([620,640],[110,110], 'blue');  plt.plot([620,620],[3,110], 'blue') 
        plt.plot([0,20],[275,275], 'cyan'); plt.plot([0,20],[390,390], 'cyan');  plt.plot([20,20],[275,390], 'cyan') 
        plt.plot([620,640],[275,275], 'green'); plt.plot([620,640],[390,390], 'green');  plt.plot([620,620],[275,390], 'green') 
        
    # invert the y-axis    
    ax.set_ylim(ax.get_ylim()[::-1])  
    plt.show()