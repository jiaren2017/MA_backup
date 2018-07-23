import numpy as np
import matplotlib.pyplot as plt
import pickle


################################################## 
######     reshaping and normalizing data    #####
################################################## 
def featureScaling(dataMat):            
    meanMat = np.mean(dataMat,axis=0)
    stdDevMat = np.std(dataMat,axis=0)
    normMat = np.divide(np.subtract(dataMat, meanMat), stdDevMat)
    return meanMat, stdDevMat, normMat

def normData(dataMat):
    dataMat = dataMat.T
#    print ("dataMat's shape: " + str(dataMat.shape))
    norm_parameters = {}
    meanMat = np.mean(dataMat,axis=1)
    stdDevMat = np.std(dataMat,axis=1)
    normMat = np.divide(np.subtract(dataMat, meanMat), stdDevMat)
    norm_parameters['mean'] = meanMat
    norm_parameters['stdDev'] = stdDevMat
    
    return norm_parameters, normMat

################################################## 
######     activation function    #####
################################################## 
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s.T  * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ
################################################## 
######     Initilizing function    #####
################################################## 
    
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
#    np.random.seed(1)
    parameters = {}                # init. dictionary
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #* 0.01  # bigger the w, faster it will learn
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters



################################################## 
######     Forward propagation module    #####
################################################## 
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "None":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = Z       # no activation function    
        activation_cache = Z                 
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache

    
    
    
def L_model_forward(X, parameters, regMode = False, foreAct = 'relu'):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    if (foreAct == 'sigmoid'):
        for l in range(1, L):
            A_prev = A 
            A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "sigmoid")  
            caches.append(cache)
#            print('forward, sigmoid')
            
    elif(foreAct == 'relu'):
        for l in range(1, L):
            A_prev = A 
            A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu") 
            caches.append(cache)
#            print('forward, relu')
            
            
    if (regMode == False):   # classification, use 'sigmoid'
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")   
        caches.append(cache)
#        print('forward final, sigmoid')
        
    else:                      # regression, use 'none'
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "None")  
        caches.append(cache)
#        print('forward final, none')
    assert(AL.shape == (1,X.shape[1]))
   
    return AL, caches



################################################## 
######     Cost function    #####
################################################## 

def compute_cost(AL, Y, regMode = False):
    m = Y.shape[1]                               # number of example
    
    if (regMode == False):    # logistic regression
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
#        print('cost for logistic')
        
    else:    # regression, compute the Mean Square Error (MSE)
        cost = np.power(AL - Y, 2)    # cost = (y_pred - y_real)^2
        cost = np.sum(cost) / (m * 2)       # get the average cost
#        print('cost for regression, mean square error')
        
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.  E.g., turns [[17]] into 17 
    assert(cost.shape == ())
    return cost


################################################## 
######     Backward propagation module    #####
################################################## 
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    #db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    db = 1./m * np.sum(dZ, axis = 1)
    db = db.reshape(b.shape)

    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db 


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "None":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, regMode = False, backAct = 'relu'):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    if (regMode == False):      # classification
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    else:                       # regression 
        dAL = (AL - Y) / m


    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    
    if (regMode == False):      # classification, use 'sigmoid'
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")   
#        print('backward first, sigmoid')
    
    else:                       # regression, use 'none'
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "None")     
#        print('backward first, none')
    
    if (backAct == 'sigmoid'):
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "sigmoid")  
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
#        print('backward, sigmoid')
        
    elif (backAct == 'relu'):
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu") 
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
#        print('backward, relu')
        
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 500, print_cost= False, plot_cost = False, regMode = False, foreAct = 'relu', backAct = 'relu'): 
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
#    print('parameters: ', parameters)
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, regMode, foreAct)
        
        # Compute cost.
        cost = compute_cost(AL, Y, regMode)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, regMode, backAct)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
 
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    if plot_cost:               # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters




def network_predict(dataMat_test, parameters_NN, norm_parameters_NN = None, regMode = False, foreAct = 'relu', stepPred= False, parameters_xNN=None, x_norm_parameters=None,  parameters_yNN=None, y_norm_parameters=None, numSteps=10):
    if (stepPred == False):                      #  just run forecast based on the test data, static
        if (norm_parameters_NN == None):
            yArr_pred, caches = L_model_forward(dataMat_test, parameters_NN, regMode, foreAct)
        else:
            dataMat_test_normed = np.divide(np.subtract(dataMat_test, norm_parameters_NN['mean']), norm_parameters_NN['stdDev'])
            yArr_pred, caches = L_model_forward(dataMat_test_normed, parameters_NN, regMode, foreAct)
        return yArr_pred
            
    else:    #  predict the steps, dynamic! 
        dataMat_pred = np.mat(np.zeros((6,numSteps)))   # features are rows vectors and samples are column vectors !!!
        dataMat_pred[0:4, 0] = dataMat_test             # assign the position and speed of test data
        for i in range(numSteps-1):
            ## calculate the acceleratex x(i), y(i);    input data should be normalized first !!!!                         
            dataMat_pred[4,i], caches = L_model_forward(np.divide(np.subtract(dataMat_pred[0:4, i], x_norm_parameters['mean']), x_norm_parameters['stdDev']), parameters_xNN, regMode, foreAct)
            dataMat_pred[5,i], caches = L_model_forward(np.divide(np.subtract(dataMat_pred[0:4, i], y_norm_parameters['mean']), y_norm_parameters['stdDev']), parameters_yNN, regMode, foreAct)
            
            # generate the data for the NEW position x(i+1), y(i+1)     new position = old position + old speed + old accelerate 
            dataMat_pred[0,i+1] = dataMat_pred[0,i] + dataMat_pred[2,i] + dataMat_pred[4,i]
            dataMat_pred[1,i+1] = dataMat_pred[1,i] + dataMat_pred[3,i] + dataMat_pred[5,i]
            
            # generate the data for the NEW speed x(i+1), y(i+1)        new speed = old speed + old accelerate    
            dataMat_pred[2,i+1] = dataMat_pred[2,i] + dataMat_pred[4,i]
            dataMat_pred[3,i+1] = dataMat_pred[3,i] + dataMat_pred[5,i]
                                                    ## input data should be normalized first !!!!         
        dataMat_pred[4,-1], caches = L_model_forward(np.divide(np.subtract(dataMat_pred[0:4, -1], x_norm_parameters['mean']), x_norm_parameters['stdDev']), parameters_xNN, regMode, foreAct)
        dataMat_pred[5,-1], caches = L_model_forward(np.divide(np.subtract(dataMat_pred[0:4, -1], y_norm_parameters['mean']), y_norm_parameters['stdDev']), parameters_yNN, regMode, foreAct)
        
        return dataMat_pred

        
        
def saveNetwork(inParameter, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inParameter, fw)
    
def loadNNetwork(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
    
    
###################################################################





