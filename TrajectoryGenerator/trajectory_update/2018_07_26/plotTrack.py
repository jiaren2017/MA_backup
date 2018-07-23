import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch

def getPosArr(dataMat):                                      # extract the position data from matrix
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
    

def plot_1234(dataMat_org, dataMat_Pred=None, origShow=False, predShow =True, fig_size = (10,10), minX_plot = None, maxX_plot = None, minY_plot = None, maxY_plot = None):
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
    
    ### plot gates 1,2,3,4
    plt.plot([0,20],[3,3], 'red'); plt.plot([0,20],[110,110], 'red');  plt.plot([20,20],[3,110], 'red') 
    plt.plot([620,640],[3,3], 'blue'); plt.plot([620,640],[110,110], 'blue');  plt.plot([620,620],[3,110], 'blue') 
    plt.plot([0,20],[275,275], 'cyan'); plt.plot([0,20],[390,390], 'cyan');  plt.plot([20,20],[275,390], 'cyan') 
    plt.plot([620,640],[275,275], 'green'); plt.plot([620,640],[390,390], 'green');  plt.plot([620,620],[275,390], 'green') 
    
    plt.show()