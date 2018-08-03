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