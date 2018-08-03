import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

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
######     preprocessing training set    #####
################################################## 

# GRADED FUNCTION: random_mini_batches
def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]  #    shuffled_Y = Y[:, permutation].reshape((4,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size "mini_batch_size" in the partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches












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
    
def softmax(Z):
#    print("\n\nZ.shape:",Z.shape)
    exp_Z = np.exp(Z)
    
#    print("\n\nZ[:,:3]\n:",Z[:,:3])
#    print("\n\nexp_Z[:,:3]\n:",exp_Z[:,:3])

    A = exp_Z / np.sum(exp_Z, axis=0) #keepdims=True)  
    
#    print("\n\nnp.sum(exp_Z[:,:3], axis=0)\n:",np.sum(exp_Z[:,:3], axis=0))
#    print("\n\nA.shape:",A.shape)
#    print("\n\nA[:,:3]\n:",A[:,:3])

    assert(math.isnan(Z[0,0]) == False)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

    
def softmax_backward(AL, Y, cache):
    Z = cache
    dZ = AL - Y
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
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])   # He initialization works well for networks with ReLU activations.
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

    
# Momentum
# Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, 
# and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.
# GRADED FUNCTION: initialize_velocity
def initialize_velocity(parameters):   
    L = len(parameters) // 2        # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
    return v

    
    
# GRADED FUNCTION: initialize_adam    
def initialize_adam(parameters) :
    L = len(parameters) // 2        # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)
    return v, s
    
    
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

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
        
    elif activation == "None":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = Z       # no activation function    
        activation_cache = Z         
        

    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache

    
    
    
def L_model_forward(X, parameters, learning_mode = "regression", foreAct = 'relu'):
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
            
            
    ## calculate the final prediction on the last output layer        
    if (learning_mode == "classification"):    # softmax
        prediction, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")   
        caches.append(cache)
#        print('forward final, softmax')
        
    elif(learning_mode == "regression"):
        prediction, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "None")  
        caches.append(cache)
        print('forward final, none')
#    print("prediction.shape:", prediction.shape)
    #assert(prediction.shape == (4, X.shape[1]))
   
    return prediction, caches



################################################## 
######     Cost function    #####
################################################## 

def compute_cost(AL, Y, learning_mode = "regression"):
    m = Y.shape[1]      # number of example
    
    if (learning_mode == "classification"):    # softmax
        assert (AL.shape == Y.shape)
#        print("learning_mode == classification")
#        print("\n\nAl[:,:3]:\n", AL[:,:3]); print("\n\nY[:,:3]:\n", Y[:,:3])
        
        cost_matrix = np.multiply(AL, Y)    ## elementweise multiply: [0.1, 0.2, 0.3, 0.4].T * [1,0,0,0].T = [0.1,0,0,0].T;   cost_matrix: [4, num_samples]
#        print("\n\npre_cost_matrix[:,:3]:\n", cost_matrix[:,:3])

        #raise NameError('')
        cost_matrix = np.sum(cost_matrix, axis=0)      ## add each column value together, so that the cost_matrix: [1, num_samples]
#        print("\n\nsum_cost_matrix[:,:3]:\n", cost_matrix[:,:3])
        cost_matrix = -np.log(cost_matrix)      ## add each column value together, so that the cost_matrix: [1, num_samples]
#        print("\n\nlog_cost_matrix[:,:3]:\n", cost_matrix[:,:3])
        cost = np.sum(cost_matrix) / m                  ## add all cost together = cross entropy
#        print("\n\nsum of cost:\n", cost)

        
    elif(learning_mode == "regression"):     # regression, compute the Mean Square Error (MSE)
        print("learning_mode == regression")
        cost = np.power(AL - Y, 2)          # cost = (y_pred - y_real)^2
        cost = np.sum(cost) / (m * 2)       # get the average cost

        
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


def linear_activation_backward(dA_or_AL, cache, activation, Y, ):
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
        dZ = relu_backward(dA_or_AL, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA_or_AL, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":        
        dZ = softmax_backward(dA_or_AL, Y, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "None":
        dZ = dA_or_AL
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, learning_mode = "regression", backAct = 'relu'):
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
    if (learning_mode == "classification"):     # classification (softmax)
        pass    ## for softmax, we can directly calculate the dz = AL - Y, e.g: AL:[0.1,0.2,0.25,0.4,0.05], y:[0,0,1,0,0] -> dz = [0.1,0.2,-0.75,0.4,0.05]
        ##dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))       ## derivative logistic regression
        
    elif(learning_mode == "regression"):        # regression 
        dAL = (AL - Y) / m


    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    
    if (learning_mode == "classification"):     # classification, use 'sigmoid'
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(AL, current_cache, activation = "softmax", Y=Y)   
#        print('backward first, sigmoid')
    
    elif(learning_mode == "regression"):        # regression, use 'none'
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "None", Y = None)     
#        print('backward first, none')
    
    
    ## calculate the back propogation from l-1 layer to first layer
    if (backAct == 'sigmoid'):
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "sigmoid", Y = None)  
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
#        print('backward, sigmoid')
        
    elif (backAct == 'relu'):
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu", Y = None) 
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
#        print('backward, relu')
        
    return grads


################################################## 
######     Update W and b    #####
################################################## 

# GRADED FUNCTION: update_parameters_with_gd
def update_parameters_with_gd(parameters, grads, learning_rate):   
#    print("check update with gd")
    L = len(parameters) // 2        # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

    
    
# GRADED FUNCTION: update_parameters_with_momentum
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate): 
    L = len(parameters) // 2        # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta*v["dW" + str(l + 1)] + (1-beta)*grads['dW' + str(l+1)]
        v["db" + str(l + 1)] = beta*v["db" + str(l + 1)] + (1-beta)*grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters['W' + str(l+1)] - learning_rate*v["dW" + str(l + 1)] 
        parameters["b" + str(l + 1)] = parameters['b' + str(l+1)] - learning_rate*v["db" + str(l + 1)] 
        
    return parameters, v   
    
    
    
# GRADED FUNCTION: update_parameters_with_adam
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
    
        #### Momentum ####
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = beta1*v["dW" + str(l + 1)] + (1-beta1)*grads['dW' + str(l+1)]
        v["db" + str(l + 1)] = beta1*v["db" + str(l + 1)] + (1-beta1)*grads['db' + str(l+1)]
        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1-(beta1)**t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1-(beta1)**t)

        #### RMSProp ####
        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        # s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)]**2)     # "**2" only works if grads["dW" + str(l+1)] is array!!!
        # s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)]**2)
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (np.multiply(grads["dW" + str(l+1)], grads["dW" + str(l+1)]))  # if grads["dW" + str(l+1)] is matrix, then use np.multiply !!!
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (np.multiply(grads["db" + str(l+1)], grads["db" + str(l+1)])) 
        
        
        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)


        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)
        
    return parameters, v, s
    
    
    
    
################################################## 
######     complete model    #####
##################################################
def L_layer_model(X, Y, layers_dims, optimizer, learning_rate = 0.0075, num_iterations = 500, enable_mini_batch = False, mini_batch_size = 64, print_cost= False, plot_cost = False, learning_mode = "regression",
                    foreAct = 'relu', backAct = 'relu', beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8): 
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    learning_mode -- regression or classification
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    print("BEGIN")
    
    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
        
    
    # Optimization loop
    for i in range(num_iterations):
        if (enable_mini_batch == True):
            minibatches = random_mini_batches(X, Y, mini_batch_size)  # minibatch: matrix_x,  matrix_y
            
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
            
                # Forward propagation
                AL, caches = L_model_forward(minibatch_X, parameters, learning_mode)

                # Compute cost.
                cost = compute_cost(AL, minibatch_Y, learning_mode)
            
                # Backward propagation.
                grads = L_model_backward(AL, minibatch_Y, caches, learning_mode, backAct)
         
                # Update parameters.
                if optimizer == "gd":
                    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
                    
        else:
            AL, caches = L_model_forward(X, parameters, learning_mode)
            cost = compute_cost(AL, Y, learning_mode)
            grads = L_model_backward(AL, Y, caches, learning_mode, backAct)
            # Update parameters.
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
     
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    if plot_cost:               # plot the cost
        plt.plot(np.squeeze(costs))
        #plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters


    
    
################################################## 
######     predict the next steps    #####
##################################################

def network_predict(dataMat_test, parameters_NN, norm_parameters_NN = None, regMode = True, foreAct = 'relu', stepPred= False, parameters_xNN=None, x_norm_parameters=None,  parameters_yNN=None, y_norm_parameters=None, numSteps=10):
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

## auto. detect the learning step numbers
def network_predict_v2(dataMat_test, parameters_NN, norm_parameters_NN = None, regMode = True, foreAct = 'relu', stepPred= False, parameters_xNN=None, x_norm_parameters=None,  parameters_yNN=None, y_norm_parameters=None, numSteps=10):
    if (stepPred == False):                      #  just run forecast based on the test data, static
        if (norm_parameters_NN == None):
            yArr_pred, _ = L_model_forward(dataMat_test, parameters_NN, regMode, foreAct)
        else:
            dataMat_test_normed = np.divide(np.subtract(dataMat_test, norm_parameters_NN['mean']), norm_parameters_NN['stdDev'])
            yArr_pred, _ = L_model_forward(dataMat_test_normed, parameters_NN, regMode, foreAct)
        return yArr_pred
            
    else:    #  predict the steps, dynamic! 
        dataMat_pred = np.mat(np.zeros((dataMat_test.shape[0]+2, numSteps+1)))   # features are rows vectors (with n steps!) and samples are column vectors !!!
        dataMat_pred[0:-2, 0] = dataMat_test             # assign the position and speed of test data
        for i in range(numSteps):
            ## copy the old steps to the next column 
            dataMat_pred[0:-6, i+1] = dataMat_pred[4:-2, i]
        
            ## calculate the acceleratex x(i), y(i);    input data should be normalized first !!!!                         
            dataMat_pred[-2,i], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, i], x_norm_parameters['mean']), x_norm_parameters['stdDev']), parameters_xNN, regMode, foreAct)
            dataMat_pred[-1,i], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, i], y_norm_parameters['mean']), y_norm_parameters['stdDev']), parameters_yNN, regMode, foreAct)
             
            # calculate the data for the NEW position x(i+1), y(i+1)     new position = old position + old speed + old accelerate 
            dataMat_pred[-6,i+1] = dataMat_pred[-6,i] + dataMat_pred[-4,i] + dataMat_pred[-2,i]
            dataMat_pred[-5,i+1] = dataMat_pred[-5,i] + dataMat_pred[-3,i] + dataMat_pred[-1,i]
            
            # calculate the data for the NEW speed x(i+1), y(i+1)        new speed = old speed + old accelerate    
            dataMat_pred[-4,i+1] = dataMat_pred[-4,i] + dataMat_pred[-2,i]
            dataMat_pred[-3,i+1] = dataMat_pred[-3,i] + dataMat_pred[-1,i]
            
        ## calculate the acceleration, input data should be normalized first !!!!         
        dataMat_pred[-2,-1], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, -1], x_norm_parameters['mean']), x_norm_parameters['stdDev']), parameters_xNN, regMode, foreAct)
        dataMat_pred[-1,-1], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, -1], y_norm_parameters['mean']), y_norm_parameters['stdDev']), parameters_yNN, regMode, foreAct)
        
        return dataMat_pred
        

def network_predict_v3(dataMat_test, parameters_NN, norm_parameters_NN = None, prediction_mode = "regression", foreAct = 'relu', stepPred= False, parameters_xNN=None, x_norm_parameters=None,  parameters_yNN=None, y_norm_parameters=None, numSteps=10):
    if (stepPred == False):                      #  just run forecast based on the test data, static
        if (norm_parameters_NN == None):
            yArr_pred, _ = L_model_forward(dataMat_test, parameters_NN, prediction_mode, foreAct)
        else:
            dataMat_test_normed = np.divide(np.subtract(dataMat_test, norm_parameters_NN['mean']), norm_parameters_NN['stdDev'])
            yArr_pred, _ = L_model_forward(dataMat_test_normed, parameters_NN, prediction_mode, foreAct)
        return yArr_pred
            
    else:    #  predict the steps, dynamic! 
        dataMat_pred = np.mat(np.zeros((dataMat_test.shape[0]+2, numSteps+1)))   # features are rows vectors (with n steps!) and samples are column vectors !!!
        dataMat_pred[0:-2, 0] = dataMat_test             # assign the position and speed of test data
        for i in range(numSteps):
            ## copy the old steps to the next column 
            dataMat_pred[0:-6, i+1] = dataMat_pred[4:-2, i]
        
            ## calculate the acceleratex x(i), y(i);    input data should be normalized first !!!!                         
            dataMat_pred[-2,i], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, i], x_norm_parameters['mean']), x_norm_parameters['stdDev']), parameters_xNN, regMode, foreAct)
            dataMat_pred[-1,i], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, i], y_norm_parameters['mean']), y_norm_parameters['stdDev']), parameters_yNN, regMode, foreAct)
             
            # calculate the data for the NEW position x(i+1), y(i+1)     new position = old position + old speed + old accelerate 
            dataMat_pred[-6,i+1] = dataMat_pred[-6,i] + dataMat_pred[-4,i] + dataMat_pred[-2,i]
            dataMat_pred[-5,i+1] = dataMat_pred[-5,i] + dataMat_pred[-3,i] + dataMat_pred[-1,i]
            
            # calculate the data for the NEW speed x(i+1), y(i+1)        new speed = old speed + old accelerate    
            dataMat_pred[-4,i+1] = dataMat_pred[-4,i] + dataMat_pred[-2,i]
            dataMat_pred[-3,i+1] = dataMat_pred[-3,i] + dataMat_pred[-1,i]
            
        ## calculate the acceleration, input data should be normalized first !!!!         
        dataMat_pred[-2,-1], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, -1], x_norm_parameters['mean']), x_norm_parameters['stdDev']), parameters_xNN, regMode, foreAct)
        dataMat_pred[-1,-1], _ = L_model_forward(np.divide(np.subtract(dataMat_pred[0:-2, -1], y_norm_parameters['mean']), y_norm_parameters['stdDev']), parameters_yNN, regMode, foreAct)
        
        return dataMat_pred


def prediction_destination(prediction_destination_set, test_destination_set):    
    assert (prediction_destination_set.shape == test_destination_set.shape)       ## prediction_destination_set: matrix [4 x num_samples]
    num_match = 0
    error_list = []
    for i in range(0, prediction_destination_set.shape[1]-1):
        maxID_in_row_predition = np.where(prediction_destination_set[:,i]==np.max(prediction_destination_set[:,i]))  ## find the row_ID of maxValue
        maxID_predition = maxID_in_row_predition[0][0]  ## extract the ID_number
        maxID_in_row_test = np.where(test_destination_set[:,i]==np.max(test_destination_set[:,i]))      ## find the row_ID of maxValue
        maxID_test = maxID_in_row_test[0][0]            ## extract the ID_number
        if (maxID_predition == maxID_test): num_match = num_match + 1
        else:   error_list.append(i)
            
        '''
        print("\n\n column_ID: ", i)
        print("\nmax_in_row_predition: ", prediction_destination_set[maxID_predition, i])
        print("\nmaxID_in_row_predition: ", maxID_predition)
        print("\nmax_in_row_test: ", test_destination_set[maxID_test, i])
        print("\nmaxID_in_row_test: ", maxID_test)
        '''
    return num_match / prediction_destination_set.shape[1], error_list
        
        
################################################## 
######     save and load network    #####
##################################################    
def saveNetwork(inParameter, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inParameter, fw)
    
def loadNNetwork(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
    
    
###################################################################





