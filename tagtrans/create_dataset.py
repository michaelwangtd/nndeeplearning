# -*- coding:utf-8 -*-

import os
import codecs
import json
from collections import OrderedDict
import numpy as np


"""
    
"""

def readListFromTxt(filePath):
    infoList = []
    if os.path.exists(filePath):
        fr = codecs.open(filePath,'r',encoding='utf-8')
        while True:
            line = fr.readline()
            if line:
                temp = line.strip()
                infoList.append(temp)
            else:
                break
        fr.close()
    return infoList



def writeList2Txt(filePath,infoList):
    if infoList:
        fw = codecs.open(filePath,'w',encoding='utf-8')
        for i in range(len(infoList)):
            outputLine = infoList[i].strip()
            fw.write(outputLine + '\n')
        fw.close()


def loadData2Json(filePath):
    '''

    '''
    jsonList = []
    if os.path.exists(filePath):
        fr = codecs.open(filePath,'r',encoding='utf-8')
        i = 1
        while True:
            line = fr.readline()
            if line:
                try:
                    temp = line.strip()
                    lineJson = json.loads(temp)
                    # print(i,type(lineJson),str(lineJson))
                    i += 1
                    jsonList.append(lineJson)
                except Exception as ex:
                    print(ex)
            else:
                break
    return jsonList


def getGroupList(indexList,mapDic):
    resultList = []
    for itemIndex in indexList:
        resultList.append(mapDic[itemIndex])
    return resultList


if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/tag_trans/'
    infoList = readListFromTxt(rootPath + 'train_tag.txt')
    print(len(infoList))

    mapDic = OrderedDict()
    for i in range(len(infoList)):
        mapDic[i] = infoList[i]

    originOrder = [i for i in range(len(infoList))]
    print(len(originOrder))
    # exit(0)
    #
    shuffleList = np.random.permutation(originOrder)
    trainIndex = shuffleList[0:int(len(shuffleList)/10*6)]
    testIndex = shuffleList[int(len(shuffleList)/10*6):int(len(shuffleList)/10*8)]
    devIndex = shuffleList[int(len(shuffleList)/10*8):]
    print(len(trainIndex))
    print(len(testIndex))
    print(len(devIndex))
    # exit(0)
    #
    trainList = getGroupList(trainIndex,mapDic)
    testList = getGroupList(testIndex,mapDic)
    devList = getGroupList(devIndex,mapDic)
    print(len(trainList))
    print(len(testList))
    print(len(devList))

    writeList2Txt(rootPath + 'train.txt',trainList)
    writeList2Txt(rootPath + 'test.txt',testList)
    writeList2Txt(rootPath + 'dev.txt',devList)


