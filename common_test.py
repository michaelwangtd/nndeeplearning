# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import codecs

np.set_printoptions(suppress=True,linewidth=200)

infoList = [1,2,3,4,5,6,7,8,9]
re = np.random.permutation(infoList)
print(re)

infoList = ['1','2','3','4']
# originOrder = [i for i in range(len(infoList))]
# print(type(originOrder),originOrder)


# test = 'LOCATION'
# print(test[0])



# fr = codecs.open('D:\workstation\\repositories\\nndeeplearning\data\\test\words.pickle','rb')
# words = pickle.load(fr)
# fr.close()
# print(type(words),words)



# print(-1/0.7)
# print(1/0.7)


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
