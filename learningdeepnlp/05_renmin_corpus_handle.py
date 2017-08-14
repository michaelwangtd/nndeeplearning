# -*- coding:utf-8 -*-

import codecs

"""
    处理人民日报语料
"""

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

if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/deepnlp/'

    inFilePath = 'renmin_199801.txt'
    outTrainFilePath = 'rm_train.txt'
    outTestFilePath = 'rm_test.txt'
    outValiFilePath = 'rm_vali.txt'

    infoList = []

    fr = codecs.open(rootPath + inFilePath,'r',encoding='utf-8')

    while True:
        line = fr.readline()
        if line:
            line = line.strip()
            if line!='':
                lineList = line.split('  ')
                lineList = lineList[1:]
                lineStr = ' '.join(lineList)
                infoList.append(lineStr)
        else:
            break

    totalNum = len(infoList)
    print(totalNum)

    writeList2Txt(rootPath + outTrainFilePath, infoList[0:int(totalNum / 10 * 6)])
    writeList2Txt(rootPath + outTestFilePath, infoList[int(totalNum / 10 * 6):int(totalNum / 10 * 8)])
    writeList2Txt(rootPath + outValiFilePath, infoList[int(totalNum / 10 * 8):])