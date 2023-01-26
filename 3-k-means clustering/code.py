import numpy as np
import matplotlib.pyplot as plt
import warnings

# 准备：生成训练集，其中k为簇的数量，total_num为样本总数
def get_trainset(k, total_num):
    # np.random.seed(0) 
    # 在实验中不打算让训练集各个簇的样本数量一致或手动指定
    # 因此使得每个簇样本数量 ~ N(total_num / k, total_num / (6 * k))   
    num_temp_lst = (np.around(np.random.normal(total_num / k, total_num / (6 * k), k-1)))
    # 为了让总样本数量固定为total_num，最后一个簇的样本数量不随机生成，
    # 指定为total_num - 之前随机生成的样本数量和
    num_lst = np.append(num_temp_lst, total_num - np.sum(num_temp_lst)).astype(int)
    
    #np.random.seed(4) 
    # 各簇的中心横坐标分布x ~ U(0,50)
    x_mean_lst = np.random.normal(0, 30, size = k)
    # 各簇的中心横坐标分布y ~ U(0,40)
    y_mean_lst = np.random.normal(0, 40, size = k)
    # 各簇的协方差均值方分布CovXY = [[sigma1, 0], [0, sigma2]]
    # 其中：sigma1 ~ N(36, 25), sigma2 ~ N(25, 25)
    CovXY = np.zeros((k, 2, 2))
    CovXY[:, 0, 0] = np.random.normal(36, 5)
    CovXY[:, 1, 1] = np.random.normal(36, 5)
    
    x = np.zeros((np.sum(num_lst), 2))
    y = np.zeros(np.sum(num_lst)).astype(int)
    
    for i in range(k):  
        temp_sum = np.sum(num_lst[:i])
        x[temp_sum : temp_sum + num_lst[i], :] = np.random.multivariate_normal([x_mean_lst[i], y_mean_lst[i]], 
                                                CovXY[i], size = num_lst[i])
        y[temp_sum : temp_sum + num_lst[i]] = i
        
    return [x, y]
    
# 绘制训练集散点图
def draw_trainset2D(trainset):
    color_lst = ['violet', 'grey', 'deepskyblue', 'greenyellow', 'pink', 'blue', 'darkorange',]
    i = 0
    for [x,y] in trainset[0]:
        plt.scatter(x, y, color = color_lst[trainset[1][i]], marker = '.')
        i += 1

# k_means算法计算分类中心和标签
def k_means(trainset, k, epsilon=1e-4):
    center = np.zeros((k, trainset[0].shape[1]))
    for i in range(k):
        center[i,:] = trainset[0][np.random.randint(0, high = trainset[0].shape[0]), :]
    while True:
        distance = np.zeros(k)
        # 根据中心重新给每个点贴分类标签
        for i in range(trainset[0].shape[0]): 
            for j in range(k):
                distance[j] = np.linalg.norm(trainset[0][i] - center[j, :])
            trainset[1][i] = np.argmin(distance)
        # 根据每个点新的标签计算它的中心
        new_center = np.zeros((k, trainset[0].shape[1]))
        count = np.zeros(k)
        # 对每个类的所有点坐标求和
        for i in range(trainset[0].shape[0]):
            new_center[int(trainset[1][i]), :] += trainset[0][i]
            count[int(trainset[1][i])] += 1
        # 对每个类的所有点坐标求平均值
        for i in range(k):
            if count[i] != 0:
                new_center[i, :] = new_center[i, :] / count[i] 
        #用差值的二范数衡量精度
        if np.linalg.norm(new_center - center) < epsilon: 
            break
        else:
            center = new_center
    return center

# 绘制聚类中心
def draw_center(point_set):
    for [x,y] in point_set:
        plt.scatter(x, y, s = 100, color = 'red', marker = '+')

# 通过计算兰德指数RI判断聚类效果
def compute_RI(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a += 1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b += 1
    RI = (a + b) / (n*(n-1)/2)
    return RI

# 通过计算轮廓系数Silhouette Coefficient判断聚类效果
def compute_SC(result_set, k):
    n = len(result_set[0])
    #sc_lst存储每个样本点的轮廓系数
    sc_lst = np.zeros(n)
    for i in range(n):
        sum_lst = np.zeros(k)
        count_lst = np.zeros(k)
        for j in range(n):
            sum_lst[result_set[1][j]] = np.linalg.norm(result_set[0][j, :] - result_set[0][i, :])
            count_lst[result_set[1][j]] += 1
        ave_lst = np.zeros(k)
        ave_lst = sum_lst / count_lst
        # 当前点i与同簇内其他点的平均距离
        a = ave_lst[result_set[1][i]]
        # 当前点i与其他各个簇内点平均距离的最小值
        b = np.min(np.delete(ave_lst, result_set[1][i]))   
        sc_lst[i] = (b - a) / max(a, b)
    return np.mean(sc_lst)


if __name__ == '__main__':
    warnings.simplefilter('error')
    #self_made_data变量指示：训练时使用自己编造的数据or从UCI下载的数据
    self_made_data = False
    
    #随机生成二维高斯分布的点作为数据集
    if self_made_data == True:
        k = 3
        total_num = 200
        trainset = get_trainset(k, total_num)
        y_true = np.zeros(total_num)
        y_true[:] = trainset[1][:]
        draw_trainset2D(trainset) 
        center_point = k_means(trainset, 5)
        draw_center(center_point)
        #因为生成的点已知标签，可以用兰德指数RI评价效果
        plt.title('k = %d, total sample number = %d, Rand Index = %.4f' 
                %(k, total_num, compute_RI(y_true,trainset[1][:] )))
        plt.show()
        
    #下载UCI的iris作为数据集
    if self_made_data == False:
        data_set = np.loadtxt('3D_spatial_network.txt', delimiter=',', encoding='utf-8', usecols=(0,1,2,3))
        np.random.shuffle(data_set)
        data_set = data_set[:1000]
        print (data_set)
        print ('完成打乱！')
        y_set = np.zeros(data_set.shape[0]).astype(int)
        trainset = [data_set, y_set]
        sc_lst = []
        for k in range(2,10):
            temp_sc = []
            # 对同一个k值连续计算20次轮廓系数取平均值
            for i in range(20):
                try:
                    k_means(trainset, k)
                    temp_sc.append(compute_SC(trainset, k))
                except RuntimeWarning:
                    pass
            #print (temp_sc)
            sc_lst.append(np.mean(temp_sc))
            plt.show()
        
        print (sc_lst)
        plt.plot(range(2,10), sc_lst, color = 'darkorange')
        plt.xlabel('K Value')
        plt.ylabel('Silhouette Coefficient')
        plt.show()
        