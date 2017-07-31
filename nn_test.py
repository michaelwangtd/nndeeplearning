# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import rnn_lstm_demo


np.set_printoptions(suppress=True,linewidth=200)




def weight_init(shape):
    '''
        这里的shape是一个list
    '''
    # 产生一个均匀分布，形状为shape的tensor
    initial = tf.random_uniform(shape,minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]), maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
    return initial


def bias_init(shape):
    '''
        shapes是一个list
    '''
    initial = tf.constant(0.01, shape=shape)
    return initial


with tf.Session() as sess:
    # output = sess.run(weight_init([5,4]))
    output = sess.run(bias_init([24]))
    print(output)





# w = rnn_lstm_demo.weight_init([10,5])
# print(type(w))
# print(w)



# arr = np.array([[1,1],[1,2],[2,2],[2,3]])
# print(type(arr))
# print('origin:',arr)
# sentenceMatrix = arr
#
# arrMean = np.mean(sentenceMatrix,axis=0)
# print('mean:',arrMean)
# centerized = sentenceMatrix - np.mean(sentenceMatrix, axis=0)
# print('centerized:',centerized)
#
# st = np.std(centerized,axis=0)
# print('st:',st)
# normalized = centerized / np.std(centerized, axis=0)
#
# print('normalizd:',normalized)



# def Standardize(sentenceMatrix):
#     pass
#
# rootPath = 'D:/workstation/repositories/nndeeplearning/data/super_smart/'
# mfc = np.load(rootPath + 'X.npy',encoding='bytes')
# art = np.load(rootPath + 'Y.npy',encoding='bytes')
#
# xMatrix = mfc[0]
# yMatrix = art[0]
# print('x',xMatrix.shape)
# print('y',yMatrix.shape)
#
# xMatrixMean = np.mean(xMatrix,axis=0)
# print(len(xMatrixMean))
# print(xMatrixMean)
#
# centerized = xMatrix - xMatrixMean
# normalized = centerized/np.std(centerized,axis=0)








# testA = ['a','b','c','d']
# testB = ['a','b','c','d']
# for a,b in zip(testA,testB):
#     print(a,b)
