# -*- coding:utf-8 -*-

import codecs
import os
import json
"""
    准备数据
"""

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


def writeList2Txt(filePath,infoList):
    if infoList:
        fw = codecs.open(filePath,'w',encoding='utf-8')
        for i in range(len(infoList)):
            outputLine = infoList[i].strip()
            fw.write(outputLine + '\n')
        fw.close()


def process(inFile,outFile):
    infoList = loadData2Json(inFile)

    resultList = []
    for info in infoList:
        # print(type(info))
        # print(info)
        # print(type(info['tokens']))
        # print(info['tokens'])
        # print(type(info['tags']))
        # print(info['tags'])
        senTag = []
        if(len(info['tokens']) == len(info['tags'])):
            zipList = zip(info['tokens'],info['tags'])
            for (token,tag) in zipList:
                tag = tag.replace('/','|')
                senTag.append(token + '/' + tag)
                # print(token,tag)
        # print(senTag)
        senTagStr = ' '.join(senTag)
        print(senTagStr)
        resultList.append(senTagStr)

    print('resultList len:',len(resultList))
    writeList2Txt(outFile,resultList)


if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/tag_trans/'

    process(rootPath + 'train_tag.json',rootPath + 'train_tag.txt')

    # process(rootPath + 'test_tag.json',rootPath + 'test_tag.txt')






