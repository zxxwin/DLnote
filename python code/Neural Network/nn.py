# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# %matplotlib inline 

np.random.seed(1) # set a seed so that the results are consistent


X, Y = load_planar_dataset() 
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral);
plt.show()


def layer_sizes(X, Y):
    n_x = X.shape[0] # 输入层单元数量
    n_h = 4          # 隐藏层单元数量，在这个模型中，我们设置成4即可
    n_y = Y.shape[0] # 输出层单元数量
    
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def forward_propagation(X, parameters):
    # 从parameters中取出参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 执行正向传播操作
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache



def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # 样本个数

    # 下一行的结果是 (1, m)的矩阵
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1 - A2)) 
    
    # 将 (1, m)的矩阵求和后取平均值
    cost = - 1/m * np.sum(logprobs) 
    
    # np.squeeze()函数对矩阵进行压缩维度，删除单维度条目，即把shape中为1的维度去掉
    # 例如： np.squeeze([[ 1.23 ]]) 可以将维度压缩，最终变成 1.23
    # 目的是确保 cost 是一个浮点数
    cost = np.squeeze(cost)     
    
    return cost



def backward_propagation(parameters, cache, X, Y):
    # 获取样本的数量
    m = X.shape[1]
    
    # 从 parameters 和 cache 中取得参数
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # 计算梯度 dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads



def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    
    # 更新参数
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters



def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    # 输入层单元数
    n_x = layer_sizes(X, Y)[0] 
    # 输出层单元数
    n_y = layer_sizes(X, Y)[2] 
    
    # 随机初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y) 
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 10000次梯度下降的迭代
    for i in range(0, num_iterations):
        # 正向传播，得到Z1、A1、Z2、A2
        A2, cache = forward_propagation(X, parameters)
        
        # 计算成本函数的值
        cost = compute_cost(A2, Y, parameters)
 
        # 反向传播，得到各个参数的梯度值
        grads = backward_propagation(parameters, cache, X, Y)
 
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        
        # 每迭代1000下就输出一次cost值
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    # 返回最终训练好的参数
    return parameters



def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions



# 利用含有4个神经元的单隐藏层的神经网络构建分类模型
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# 可视化分类结果
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# 输出准确率
predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')