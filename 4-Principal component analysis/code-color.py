import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


# 对样本点trainset绕原点进行二维顺时针旋转angle弧度
def rotate_2D(trainset, angle = 0):
    x = trainset[:, 0]
    y = trainset[:, 1]
    Rotate_trainset = np.zeros((trainset.shape))
    Rotate_trainset[:, 0] = x * math.cos(angle) + y * math.sin(angle)
    Rotate_trainset[:, 1] = y * math.cos(angle) - x * math.sin(angle)
    return Rotate_trainset


# 对样本点trainset绕axis轴三维旋转angle弧度
def rotate_3D(trainset, axis, angle = 0):
    if axis == 'x':
        rotate_matrix = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    elif axis == 'y':
        rotate_matrix = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    elif axis == 'z':
        rotate_matrix = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    else:
        assert False
    return np.dot(rotate_matrix, trainset.T).T


# 准备：其中dimension为样本原始的维度，为了可视化demension只取2或3,num为样本点数量
def get_trainset(dimension, num):
    if dimension == 2:
        mean = np.array([0, 0])
        covXY = np.array([[1, 0], [0, 0.01]])
        trainset = np.random.multivariate_normal(mean, covXY, size = num)
        trainset = rotate_2D(trainset, math.pi / 6)
        return trainset
    
    elif dimension == 3:
        mean = np.array([0, 0, 0])
        cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
        trainset = np.random.multivariate_normal(mean, cov, size = num)
        trainset = rotate_3D(trainset, 'z', math.pi / 3)
        trainset = rotate_3D(trainset, 'y', math.pi / 4)
        return trainset
        
        
# 绘制训练集散点图
def draw(trainset, result_set, color1, color2):
    if (trainset.shape[1] == 2):
        for [x,y] in trainset:
            plt.scatter(x, y, color = color1, marker = '.')
            plt.axis('equal')
        for [x,y] in result_set:
            plt.scatter(x, y, color = color2, marker = '+')
            plt.axis('equal')  
        
    elif (trainset.shape[1] == 3):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for [x,y,z] in trainset:
            ax.scatter(x, y, z, s = 20, c = color1, depthshade=True)
        for [x,y,z] in result_set:
            ax.scatter(x, y, z, s = 20, c = color2, depthshade=True)
    else:
        assert False
    plt.title('total sample number = %d' %(trainset.shape[0]))
    plt.show() 


# 将训练集trainset从D维降到k维，返回重构之后的数据矩阵（N * D）
def PCA(trainset, k):
    trainset = trainset.T
    dimension = trainset.shape[0]
    mean = np.mean(trainset, axis = 1)
    norm_set = np.zeros(trainset.shape)
    for i in range(dimension):
        # 零均值化后得到norm_set (D*N)
        norm_set[i] = trainset[i] - mean[i]
    # 求出协方差矩阵
    covMatrix = np.dot(norm_set, norm_set.T)
    # 对协方差矩阵covMatrix(D*D)求特征值和特征向量
    # eigenVectors为D*D矩阵，每一列对应一个特征向量
    eigenValues, eigenVectors = np.linalg.eig(covMatrix)
    # 特征值按升序排序，并返回排序后的索引
    eigValIndex = np.argsort(eigenValues)
    # 取前k个特征值对应的特征向量(D*k)
    KEigenVector = eigenVectors[:, eigValIndex[:-(k + 1):-1]]
    # numpy.linalg.eig计算的特征向量在计算的时候是以复数的形式运算的
    # 算法在收敛时，虚部可能还没有完全收敛到0，可以对其保留实部
    KEigenVector = np.real(KEigenVector)
    # 计算降维后的数据(K*N)
    reduce_tmp_set = np.dot(KEigenVector.T, norm_set)
    # 重构之后的数据
    result_set = np.zeros(trainset.shape) 
    for i in range(dimension):
        # 重构的数据 = 重构后的各方向标量大小 * 单位方向向量 + 该方向均值
        result_set[i] = np.dot(KEigenVector[i], reduce_tmp_set) + mean[i]
    return result_set.T


# 逐一读取image文件夹中的图片，并将RGB像素值存储到trainset(16, w * h, 3)中
def read_file_to_trainset(path):
    trainset = []
    img_folder = path
    img_list = [os.path.join(nm) for nm in os.listdir(img_folder)]
    for i in img_list:
        now_path = path
        now_path = os.path.join(now_path, i)
        img = cv2.imread(now_path)
        # 为了能够快速运行得到结果，对原始图片进行压缩
        size = np.array([img.shape[0]/4, img.shape[1]/4]).astype(int)
        img = cv2.resize(img, size) 
        img_reshape = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
        trainset.append(img_reshape)
    return np.array(trainset), img.shape


# 显示trainset中前100张照片
def show_some_figure(trainset, img_shape):
    trainset = trainset[:100]
    fig, ax=plt.subplots(nrows=10, ncols=10, figsize=(10,12))
    for i in range(100):
        img = trainset[i].reshape(img_shape[0], img_shape[1], img_shape[2])
        #imshow默认信息格式为BGR，读取的是RGB需要转换
        b, g, r = cv2.split(img)
        img_new = cv2.merge([r,g,b])
        ax[i//10, i%10].imshow(img_new)
        ax[i//10, i%10].set_xticks([])
        ax[i//10, i%10].set_yticks([])
    plt.show()


# 计算前num张图片的信噪比、平均信噪比
def compute_PSNR(trainset, result_set, num):
    MSE = np.zeros(num)
    PSNR = np.zeros(num)
    for i in range(num):
        MSE[i] += np.mean(((trainset[i, :, :].reshape(-1,3) - result_set[i, :, :].reshape(-1,3))) ** 2 )
        PSNR[i] = 20 * np.log10(255 / np.sqrt(MSE[i]))
    return PSNR, np.mean(PSNR)


# 将trainset中图片的R/G/B值通过PCA分别降维到K维，返回值格式与trainset相同
def RGB_PAC(trainset, K):
    img_new = np.zeros(trainset.shape)
    for i in range(3):
        img_new[:, :, i] = PCA(trainset[:, :, i], K)
    return img_new


# 将trainset降维到分别降维到K维，并计算对应前num张图片的平均PSNR
# 返回对应K的PSNR列表、并绘制折线图
def get_different_PSNR(trainset, num, K_range):
    PSNR_lst = []
    for K in K_range:
        img_new = RGB_PAC(trainset, K)
        PSNR, mean = compute_PSNR(trainset, img_new, num)
        PSNR_lst.append(mean)
        
    plt.plot(K_range, PSNR_lst, color = 'darkorange')
    plt.xlabel('K value')
    plt.ylabel('PSNR')
    plt.show()
    return PSNR_lst
    
    
if __name__ == '__main__':
    # self_made_data变量指示：训练时使用自己编造的数据or人脸的图片数据
    self_made_data = False
    
    # 随机生成二维高斯分布的点作为数据集
    if self_made_data == True:
        dimension = 3
        K = 1   # 降维到k维
        num = 100
        trainset = get_trainset(dimension, num)
        print (trainset)
        draw(trainset, PCA(trainset, K), 'blue', 'red')
        
    # 下载人脸的图片数据作为数据集
    if self_made_data == False:
        trainset, img_shape = read_file_to_trainset('more_image')
        show_some_figure(trainset[:100], img_shape)
        print (img_shape)
        K = 500
        img_new = RGB_PAC(trainset, K)   
        # 因为PCA后RGB值都是浮点数，除以255是为了令imshow按照浮点数读取   
        show_some_figure(np.clip(img_new/255, 0, 1), img_shape)      
        get_different_PSNR(trainset, 10, np.linspace(0,100,21).astype(int))
        