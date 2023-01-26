import numpy as np
import matplotlib.pyplot as plt
import warnings

#准备：数据集及精确解
def get_trainset(N, start, end):
    # 噪点横坐标~U(strat, end)
    x = sorted(np.random.rand(N) * (end - start) + start)
	# 噪声~N(0,1/16)
    y = np.sin(x) + np.random.randn(N) / 4
    return np.array([x,y]).T
 
def draw_exact_solution(start, end):
    x = np.linspace(start, end, num = round(100*(end-start)))
    y = np.sin(x)
    plt.plot(x, y, label = 'y = sin(x)')

#1.无正则项的最小二乘法拟合
def fit_GLS(trainset, m):
    X = np.array([trainset[:, 0] ** i for i in range(m + 1)]).T
    Y = trainset[:, 1]
    # 系数向量 W = (X^T * X)^(-1) * X^T * Y
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

#2.有正则项的回归，其中l为正则项系数lambda
def fit_ridge(trainset, m = 5, l = 0.5):
    X = np.array([trainset[:, 0] ** i for i in range(m + 1)]).T
    Y = trainset[:, 1]
    # 系数向量 W = (X^T * X + lambda * E)^(-1) * X^T * Y
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + l * np.eye(m + 1)), X.T), Y)

#3.梯度下降法，目标函数是凸函数可以固定学习率
def fit_GD(trainset, m = 3,  lr = 0.01, e = 1e-4):
    global iter_count
    # 随机在[0,1)内生成初始值
    w = np.random.rand(m + 1) 
    
    N = len(trainset)
    X = np.array([trainset[:, 0] ** i for i in range(len(w))]).T
    Y = trainset[:, 1]

    while True:
        iter_count += 1
        #均方误差更容易收敛
        H_X = np.dot(X, w)
        grad = 2 * np.dot(X.T, H_X - Y) / N
        w -= lr * grad
        if np.linalg.norm(grad, ord=2) < e: 
            return w
            break
        
#4.共轭梯度法，其中l为正则项参数
def fit_CG(trainset, m = 5, l = 0, e = 1e-6):
    global iter_count
    X = np.array([trainset[:, 0] ** i for i in range(m + 1)]).T
    A = np.dot(X.T, X) + l * np.eye(m + 1)
    #用特征值均大于0断言正定
    assert np.all(np.linalg.eigvals(A) > 0), '系数矩阵非正定'
    b = np.dot(X.T, trainset[:, 1])
    w = np.random.rand(m + 1)

    # 初始化参数
    d = r = b - np.dot(A, w)
    r0 = r
    while True:
        iter_count += 1
        alpha = np.dot(r.T, r) / np.dot(np.dot(d, A), d)
        w += alpha * d
        new_r = r - alpha * np.dot(A, d)
        beta = np.dot(new_r.T, new_r) / np.dot(r.T, r)
        d = beta * d + new_r
        r = new_r
        if np.linalg.norm(r) / np.linalg.norm(r0) < e:
            break
    return w
  

def draw(trainset, w, color, label):
    X = np.array([trainset[:, 0] ** i for i in range(len(w))]).T
    Y = np.dot(X, w)
    plt.plot(trainset[:, 0], Y, c = color, label = label)

if __name__ == '__main__':
    iter_count = 0
    #拟合在[-2Π，2Π]上进行
    start = (-1) * np.pi
    end  = 1 * np.pi
    trainset = get_trainset(100, start, end)
    # 绘制训练集散点图
    for [x, y] in trainset:
        plt.scatter(x, y, color = 'darkgray', marker = '.')
        
    draw_exact_solution(start, end)
    w = fit_CG(trainset, m = 7, l = 0.02, e = 1e-6)
    #这个语句只用来判断梯度下降法或共轭梯度法的迭代速度
    print("迭代了{}次".format(iter_count))
    
    draw(trainset, w, color = 'dimgrey', label = 'CG')
    plt.legend()
    plt.show()



