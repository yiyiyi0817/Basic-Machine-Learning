import numpy as np
import matplotlib.pyplot as plt

#准备：生成训练集，其中covXY为0时符合朴素贝叶斯假设
def get_trainset(pos_num, neg_num, covXY = 0):
    sigma = [4, 2]    #方差只与X向量的第几维有关，而与Y无关（不同标签的sigma都相等）
    pos_mean = [6, 2]     #正例x,y方向上的均值μ
    neg_mean = [2, 6]     #反例x,y方向上的均值μ
    
    x = np.zeros((pos_num + neg_num, 2))  
    y = np.zeros(pos_num + neg_num)
    
    #在训练集数组中：正例在前，反(0)例在后
    x[:pos_num, :] = np.random.multivariate_normal(pos_mean, [[sigma[0], covXY],
                                                    [covXY, sigma[1]]], size=pos_num)
    x[pos_num:, :] = np.random.multivariate_normal(neg_mean, [[sigma[0], covXY], 
                                                    [covXY, sigma[1]]], size=neg_num) 
    y[:pos_num] = 1
    y[pos_num:] = 0
    #print (x)
    #print (y)
    return [x,y]

# 绘制训练集散点图
def draw_trainset2D(trainset, pos_num):
    for [x,y] in trainset[:pos_num]:
        plt.scatter(x, y, color = 'orangered', marker = '.')
    for [x,y] in trainset[pos_num:]:  
        plt.scatter(x, y, color = 'royalblue', marker = '+')   

def read_dataset(path):
    file = np.loadtxt(path, delimiter=',',
                          encoding = 'utf-16', dtype=int, skiprows=1)
    x = file[:,1:]
    y = file[:,0]
    #print (x)
    #print (y)
    return [x,y]

#求sigmoid函数
def sigmoid(x):
    if x >= 0:
        return 1.0/(1 + np.exp(-x))
    else:
        return np.exp(x)/(1 + np.exp(x))

#求代价函数关于w向量的的一阶导数（结果为向量）,l为正则项系数
def derivative_1(trainset, w, l, dimension):
    result = np.zeros(dimension)
    #x = trainset[0] (不含常数项), Y = trainset[1]
    for i in range(len(trainset[0])):
        #X在x上加入常数项，方便与w相乘
        X = np.append(trainset[0][i], 1)
        result = result + X * (trainset[1][i] - sigmoid(np.dot(w, X)))
    
    return -result + l * w

#求代价函数关于w向量的的二阶导数（结果为矩阵）,l为正则项系数
def derivative_2(trainset, w, l, dimension):
    result = np.eye(dimension) * l
    
    for i in range(len(trainset[0])):
        X = np.append(trainset[0][i], 1)
        temp = sigmoid(np.dot(w, X))
        result = result + np.dot(X[:, None], X[None, :]) * temp * (1-temp)
        #print (np.dot(X[:, None], X[None, :]))
    return result

def compute_loss(trainset, w, l, dimension):
    loss = 0
    for i in range(len(trainset[0])):
        X = np.append(trainset[0][i], 1)
        loss += -trainset[1][i] * np.dot(X, w) + np.log(1+np.exp(np.dot(X, w)))
    return loss + l/2 * np.dot(w.T, w)

#牛顿迭代法求解w
def Newton(trainset, l, epsilon):
    #w的维数 = x（trainset[0]）的维数 + 1
    dimension = len(trainset[0][0]) + 1
    w = np.zeros(dimension)
    loss_lst = []
    while True:
            gradient = derivative_1(trainset, w, l, dimension)
            #print (gradient)
            # 满足精度要求(梯度的范数足够小时)，即可退出迭代
            if np.linalg.norm(gradient) <= epsilon:
                break
            # 使用迭代公式进行下一次迭代
            #print (derivative_2(trainset, w, l, dimension))
            H_inv = np.linalg.inv(derivative_2(trainset, w, l, dimension))
            w = w - np.dot(H_inv, gradient)
            #print(H_inv)
            loss_lst.append(compute_loss(trainset, w, l, dimension))
    return [w, loss_lst]


def draw_classifier2D(w, pos_num, neg_num, Accuracy):
    x = np.linspace(-5,15,10)
    y = ( -w[2] - w[0] * x) / w[1]
    plt.plot(x, y, label = 'wTX = 0', color = 'darkorchid')
    plt.title('positive number=%d, negative number=%d, Accuracy=%.4f' 
              %(pos_num, neg_num, Accuracy))
    plt.legend()
    plt.show()
    
def draw_loss(loss_lst):
    plt.plot(loss_lst, color = 'darkorange')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('LossDecline')
    plt.show()

def get_accuracy(trainset, w):
    correct_num = 0
    for i in range(len(trainset[0])):
        X = np.append(trainset[0][i], 1)
        if (np.dot(X, w) > 0 and trainset[1][i] == 1):
            correct_num += 1
        elif (np.dot(X, w) < 0 and trainset[1][i] == 0):
            correct_num += 1 
    return correct_num/len(trainset[0])

if __name__ == '__main__':
    #self_made_data变量指示：训练时使用自己编造的数据or从UCI下载的数据
    self_made_data = False
    
    #随机生成二维高斯分布的点作为数据集
    if self_made_data == True:
        pos_num = 600
        neg_num = 600
        trainset = get_trainset(pos_num, neg_num, covXY = -3)
        #trainset[0]是X部分，[1]是Y部分
        draw_trainset2D(trainset[0], pos_num)  
        w, loss_lst = Newton(trainset, 0.1, 0.000001)
        draw_classifier2D(w, pos_num, neg_num, get_accuracy(trainset, w))
        draw_loss(loss_lst)

    #下载UCI的幸福感测量作为数据集
    if self_made_data == False:
        data_set = read_dataset('实验2\SomervilleHappinessSurvey2015.csv')
        split = 80
        #前split组数据作为训练集
        train_set = [data_set[0][:split], data_set[1][:split]]
        #后(143-split)组数据作为测试集
        test_set = [data_set[0][split:], data_set[1][split:]]
        w, loss_lst = Newton(data_set, 0, 0.000001)
        print ('accuracy = %.4f' %get_accuracy(data_set, w))
        draw_loss(loss_lst)