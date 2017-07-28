# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def dataPreHandle(X,Y):
    '''
        数据预处理
    '''
    resultList = []
    # 初始化列向量的维度
    colDimension = X[0].shape[1]

    for xItem,yItem in zip(X,Y):
        pass


def Standardize(sentenceMatrix):
    '''
        矩阵标准化处理
    '''

    centerized = sentenceMatrix - np.mean(sentenceMatrix,axis=0)

    normalized = centerized/np.std(centerized,axis=0)

    return normalized




if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/super_smart/'

    # 以numpy的方式加载数据，注意这里的编码方式
    mfc = np.load(rootPath + 'X.npy',encoding='bytes')
    art = np.load(rootPath + 'Y.npy',encoding='bytes')
    print('mfc shape:',mfc.shape)
    print('art shape:',art.shape)

    # 总的数据个数
    totalSamples = len(mfc)
    # print('total samples:',totalSamples)
    # 设置验证集个数
    validationSet = 0.2

    '''
        注意数据的输入形式:
        其中输入数据的形状是[n_samples, n_steps, D_input]
        其中输出数据的形状是[n_samples, D_output]
    '''

    # 进行数据处理
    data = dataPreHandle(mfc,art)







