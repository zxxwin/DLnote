import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)

# 实现几个辅助函数
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

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

def sigmoid_backward(dA, cache):    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        # 我们将随机初始化的 W 乘上 np.sqrt(1 / layer_dims[l-1]) 
        # 目的是使得最终 Z 的值不至于过大, 导致落在激活函数斜率较小的部分
        # 斜率太小,梯度下降的速度就慢了
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A_prev, W, b):  
    Z = W.dot(A_prev) + b

    # cache 存放当前层线性部分的参数 "A_prev", "W" 和 "b"
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    # A_prev 是前一层神经元的输出.
    # activation 是使用的激活函数
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    # linear_cache指的是l层对应的输入A[l -1], 以及W[l], b[l]；
    # activation_cache 指的是 Z；
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    # 获取神经网络层数, // 表示整除
    L = len(parameters) // 2                  
    
    # 实现 [LINEAR -> RELU]*(L-1). 
    # 将前 L-1 层的缓存 "cache" 添加到列表 "caches" 中.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # 实现 LINEAR -> SIGMOID
    # 将第 L 层的缓存 "cache" 添加到列表 "caches" 中.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, Y, parameters, lambd = 0.7):    
    m = Y.shape[1]
    
    # 分析矩阵的维数, 可以得知 Y 与 AL 是相同位置元素相乘的关系
    cross_entropy_cost = - 1/m * np.sum( np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log( 1 - AL )) )

    L2_regularization_cost = 0

    L = len(parameters) // 2
    for l in range(0, L):
        L2_regularization_cost += np.sum(np.square(parameters["W" + str(l+1)]))

    L2_regularization_cost *= lambd/(2 * m)

    # 或者可以这么算 cost:
    # cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    # 还要确保计算出来的 cost 是一个数, 而不是矩阵
    # cost = np.squeeze(cost)    
    
    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def linear_backward(dZ, cache, lambd):
    # 取出当前层的正向传播时缓存的数据
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T) + lambd/m * W
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# activation 用于指定当前层使用的激活函数
def linear_activation_backward(dA, cache, activation, lambd):
    # linear_cache指的是l层对应的输入A[l -1], 以及W[l], b[l]
    # activation_cache 指的是 Z；
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db

# 形参 caches 由 L 个 cache 组成, 下标为 0 到 L-1 分别对应 1 到 L 层
# 每个 cache 为 (linear_cache, activation_cache)
# linear_cache指的是l层对应的输入A[l -1], 以及W[l], b[l]；
# activation_cache 指的是 Z；
def L_model_backward(AL, Y, caches, lambd = 0.7):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # 第 L 层的反向传播
    dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)

    
    # l 从第 L-1 层遍历至第 1 层, 完成第 l 层的反向传播:
    for l in reversed(range(1, L)): 
        # 注意 caches 的下标是从0到L-1, 第 l 层的缓存为 caches[l - 1]
        current_cache = caches[l-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l)], current_cache, "relu", lambd)
        grads["dA" + str(l - 1)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp
       
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    costs = []
    
    # 初始化参数
    parameters = initialize_parameters_deep(layers_dims)
    # lambd = 0.6
    lambd = 0.65
    
    for i in range(0, num_iterations):

        # 正向传播
        AL, caches = L_model_forward(X, parameters)
        
        # 计算成本函数值
        cost = compute_cost(AL, Y, parameters, lambd)
    
        # 反向传播
        grads = L_model_backward(AL, Y, caches, lambd)
 
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # 每迭代100次打印损一次成本函数的值
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # 将成本函数值的变化用图表示出来
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def predict(X, y, parameters):
    m = X.shape[1]
    
    probas, caches = L_model_forward(X, parameters)

    p = (probas > 0.5)

    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p



# 载入数据
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape 操作, 让数据集合变成 (12288, m) 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# 标准化输入数据, 使之均介于 0 到 1 之间
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# 5 层的模型 
# layers_dims = [12288, 20, 7, 5, 1] 
layers_dims = [12288, 20, 7, 1] 


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 13000, print_cost = True)

# 对训练集的预测
pred_train = predict(train_x, train_y, parameters)

# 对测试集的预测
pred_test = predict(test_x, test_y, parameters)