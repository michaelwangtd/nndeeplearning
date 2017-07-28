# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt



if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/super_smart/'

    # 以numpy的方式加载数据，注意这里的编码方式
    mfc = np.load(rootPath + 'X.npy',encoding='bytes')
    art = np.load(rootPath + 'Y.npy',encoding='bytes')
    print('mfc shape:',mfc.shape)
    print('art shape:',art.shape)
