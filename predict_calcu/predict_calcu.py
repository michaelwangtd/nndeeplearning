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


def  classifyE1AndE2(totalList):
    E1List = []
    E2List = []
    for item in totalList:
        tempTag = item.split('__')[-1]
        if '1' in tempTag:
            E1List.append(item)
        if '2' in tempTag:
            E2List.append(item)
    return E1List,E2List


if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/predict_calcu/'

    testInfoList = readListFromTxt(rootPath + 'test_tag.txt')
    resultInfoList = readListFromTxt(rootPath + 'result_tag.txt')
    print('测试句子总数：',len(testInfoList))
    # print(len(resultInfoList))

    #
    totalPredictRight = 0
    totalPredictRightList = []
    totalPredict = 0
    totalPredictList = []
    totalRight = 0
    totalRightList = []

    for i in range(len(testInfoList)):
        # print(i)
        testSentence = testInfoList[i]
        resultSentence = resultInfoList[i]

        testSenList = testSentence.split(' ')
        resultSenList = resultSentence.split(' ')
        # print(len(testSenList))
        # print(len(resultSenList))
        if len(testSenList)==len(resultSenList):
            # 1
            for item in testSenList:
                if item.split('/')[-1]!='O':
                    totalRight += 1
                    totalRightList.append(item.split('/')[-1])
                    # if 'capuchinas' in item:
                    #     print(item)
                    #     exit(0)
            # 2
            for item in resultSenList:
                if item.split('/')[1]!='O':
                    totalPredict += 1
                    totalPredictList.append(item.split('/')[1])
            # 3
            for j in range(len(testSenList)):
                if testSenList[j].split('/')[-1]!='O':
                    tag = testSenList[j].split('/')[-1]
                    if resultSenList[j].split('/')[0] == testSenList[j].split('/')[0] \
                        and resultSenList[j].split('/')[1] == tag:
                        totalPredictRight += 1
                        totalPredictRightList.append(resultSenList[j].split('/')[1])
    # print('total predict right num（TP）:',totalPredictRight,len(totalPredictRightList))
    print('total predict right num（TP）:',totalPredictRight)
    # print('total predict num（TP+FP）:',totalPredict,len(totalPredictList))
    print('total predict num（TP+FP）:',totalPredict)
    # print('total right num（TP+FN）:',totalRight,len(totalRightList))
    print('total right num（TP+FN）:',totalRight)

    print('----------------')
    E1_E2_P = totalPredictRight / totalPredict
    E1_E2_R = totalPredictRight / totalRight
    E1_E2_F1 = (2 * E1_E2_P * E1_E2_R) / (E1_E2_P + E1_E2_R)

    print('E1_E2_Precision:',str(round(E1_E2_P,4)*100)+'%')
    print('E1_E2_Recall:',str(round(E1_E2_R,4)*100)+'%')
    print('E1_E2_F:',str(round(E1_E2_F1,4)*100)+'%')


    # for item in totalPredictRightList:
    #     print(item)

    totalPredictRight_E1_List,totalPredictRight_E2_List = classifyE1AndE2(totalPredictRightList)
    # print(len(totalPredictRight_E1_List),len(totalPredictRight_E2_List))
    totalPredict_E1_List,totalPredict_E2_List = classifyE1AndE2(totalPredictList)
    # print(len(totalPredict_E1_List),len(totalPredict_E2_List))
    totalRight_E1_List,totalRight_E2_List = classifyE1AndE2(totalRightList)
    # print(len(totalRight_E1_List),len(totalRight_E2_List))


    # for item in totalRight_E2_List:
    #     print(item)


    print('---------------------')
    E1_P = int(len(totalPredictRight_E1_List)) / int(len(totalPredict_E1_List))
    E1_R = int(len(totalPredictRight_E1_List)) / int(len(totalRight_E1_List))
    E1_F = (2 * E1_P * E1_R) / (E1_P + E1_R)
    # print(round(E1_R,4))
    # print(round(round(E1_R,4)*100,2))
    print('E1_Precision:',str(round(E1_P,4)*100)+'%')
    print('E1_Recall:',str(round(round(E1_R,4)*100,2))+'%')
    print('E1_F:',str(round(E1_F,4)*100)+'%')

    print('---------------------')
    E2_P = int(len(totalPredictRight_E2_List)) / int(len(totalPredict_E2_List))
    E2_R = int(len(totalPredictRight_E2_List)) / int(len(totalRight_E2_List))
    E2_F = (2 * E2_P * E2_R) / (E2_P + E2_R)
    print('E2_Precision:',str(round(E2_P,4)*100)+'%')
    print('E2_Recall:',str(round(E2_R,4)*100)+'%')
    print('E2_F:',str(round(E2_F,4)*100)+'%')
