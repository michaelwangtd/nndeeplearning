# -*- coding:utf-8 -*-

import os
import codecs

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

if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/predict_calcu/'

    testInfoList = readListFromTxt(rootPath + 'test_tag.txt')
    resultInfoList = readListFromTxt(rootPath + 'result_tag.txt')
    sentenceList = readListFromTxt(rootPath + 'sentence_test.txt')
    # print(len(testInfoList),len(resultInfoList),len(sentenceList))
    # exit(0)
    for i in range(len(testInfoList)):
        # print(i)
        testSentence = testInfoList[i]
        resultSentence = resultInfoList[i]

        testSenList = testSentence.split(' ')
        resultSenList = resultSentence.split(' ')

        testTagedList = []
        resultTagedList = []
        if len(testSenList) == len(resultSenList):

            for item in testSenList:
                if item.split('/')[-1] != 'O':
                    testTagedList.append(item)

            for item in resultSenList:
                if item.split('/')[-1] != 'O':
                    resultTagedList.append(item)
        print(sentenceList[i])
        print(testTagedList,'   ',resultTagedList)