# -*- coding:utf-8 -*-

import os
import codecs
import json
from collections import OrderedDict
import numpy as np

"""
    利用“洗牌”算法，生成训练数据
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
            if isinstance(infoList[i],list):
                outputLine = ','.join(infoList[i]).strip()
            elif isinstance(infoList[i],str):
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

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/joint_extr/'
    infoList = readListFromTxt(rootPath + 'sentence_tags.txt')
    print(len(infoList))

    mapDic = OrderedDict()
    for i in range(len(infoList)):
        mapDic[i] = infoList[i]

    originOrder = [i for i in range(len(infoList))]
    print(len(originOrder))

    #
    shuffleList = np.random.permutation(originOrder)
    trainIndex = shuffleList[0:int(len(shuffleList)/10*6)]
    testIndex = shuffleList[int(len(shuffleList)/10*6):int(len(shuffleList)/10*8)]
    devIndex = shuffleList[int(len(shuffleList)/10*8):]
    print(len(trainIndex))
    print(len(testIndex))
    print(len(devIndex))

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


