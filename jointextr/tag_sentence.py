# -*- coding:utf-8 -*-

import json
import os
import codecs
from collections import OrderedDict


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


def getEntityAndRelationTupleList(relDicList):

    tupleList = []
    tempDic = dict()
    for relDic in relDicList:
        # print(type(relDic),relDic)
        key = relDic['em1Text'] + '|' + relDic['em2Text']
        if key not in tempDic.keys():
            tempDic[key] = relDic
            tupleList.append((relDic['em1Text'],relDic['em2Text'],relDic['label']))
    return tupleList


def getEntityQueryDic(enMenDicList):

    # queryDic = dict()
    queryDic = OrderedDict()
    for enMenDic in enMenDicList:
        queryDic[enMenDic['text']] = enMenDic
    return queryDic


# def getAppetizerList(enRelTuple,entityQueryDic):
#
#     middleTag = getMiddleRelationTag(enRelTuple,entityQueryDic)
#     print(middleTag)
#
#     entitySequenceList = getSequenceEntityList(enRelTuple,entityQueryDic)
#     print(entitySequenceList)


def getMiddleRelationTag(enRelTuple,entityQueryDic):
    label_1 = entityQueryDic[enRelTuple[0]]['label']
    label_2 = entityQueryDic[enRelTuple[1]]['label']
    label_3 = enRelTuple[2].split('/')[-1]
    return label_1[0] + '-' + label_2[1] + '-' + label_3
    # return label_1 + '-' + label_2 + '-' + label_3


def getSequenceEntityList(enRelTuple,entityQueryDic):

    # print(type(entityQueryDic[enRelTuple[0]]['start']))
    if(entityQueryDic[enRelTuple[0]]['start'] < entityQueryDic[enRelTuple[1]]['start']):
        return [enRelTuple[0] + '|1',enRelTuple[1] + '|2']
    else:
        return [enRelTuple[1] + '|1',enRelTuple[0] + '|2']


def getEntityRelationStrList(entityQueryDic):

    enRelStrList = []

    entityOrderedList = []
    for k,v in entityQueryDic.items():
        entityOrderedList.append(v['text'])
    # print(entityOrderedList)
    for i in range(0,len(entityOrderedList)-1):
        enRelStrList.append(entityOrderedList[i] + entityOrderedList[i+1])
    # print(enRelStrList)
    return enRelStrList


def getEntitySBEITagList(entity):
    sbeiEnList = []

    enSplitList = entity.split(' ')
    if (len(enSplitList) == 1):
        sbeiEnList.append('S' + '|' + enSplitList[0])
    elif (len(enSplitList) == 2):
        sbeiEnList.append('B' + '|' + enSplitList[0])
        sbeiEnList.append('E' + '|' + enSplitList[1])
    elif (len(enSplitList) > 2):
        sbeiEnList.append('B' + '|' + enSplitList[0])
        sbeiEnList.append('E' + '|' + enSplitList[-1])
        for i in range(1,len(enSplitList)-1):
            sbeiEnList.append('I' + '|' + enSplitList[i])

    return sbeiEnList




if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/joint_extr/'
    # jsonList = loadData2Json(rootPath + 'train_test.json')
    jsonList = loadData2Json(rootPath + 'train.json')
    print(len(jsonList))


    outputList = []
    relationTagSetList = []

    i = 1
    j = 1
    for line in jsonList:

        sentence = line['sentText'].strip()

        # 整理实体和关系
        entityAndRelationTupleList = getEntityAndRelationTupleList(line['relationMentions'])
        # print(entityAndRelationTupleList)

        # 获取实体查询字典
        entityQueryDic = getEntityQueryDic(line['entityMentions'])

        preTaggedTupleList = []
        for enRelTuple in entityAndRelationTupleList:
            try:
                middleTag = getMiddleRelationTag(enRelTuple, entityQueryDic)
                # print(middleTag)
            except Exception as e:
                # print(i,e)
                i += 1
                continue

            entitySequenceList = getSequenceEntityList(enRelTuple, entityQueryDic)
            # print(entitySequenceList)

            # 生成序列标签后再筛选，筛选放在for外面进行
            # for item in entitySequenceList:
            #     preTaggedList.append(middleTag + '|' + item)
            preTaggedTupleList.append((middleTag + '|' + entitySequenceList[0],middleTag + '|' + entitySequenceList[1]))

        # print(preTaggedTupleList)

        enRelStrList = getEntityRelationStrList(entityQueryDic)
        # print(enRelStrList)

        # 初步筛选已经在enRelStrList中的候选关系
        preTaggedTupleListTwo = []
        for tupleItem in preTaggedTupleList:
            tempEnStr = tupleItem[0].split('|')[1] + tupleItem[1].split('|')[1]
            if tempEnStr in enRelStrList:
                preTaggedTupleListTwo.append(tupleItem)

        # print(preTaggedTupleListTwo)

        preTaggedTupleListThree = []
        doorList = []
        # 从preTaggedTupleListTwo中筛选出不重复的元组
        for tupleItem in preTaggedTupleListTwo:
            if(tupleItem[0].split('|')[1] not in doorList and tupleItem[1].split('|')[1] not in doorList):
                doorList.append(tupleItem[0].split('|')[1])
                doorList.append(tupleItem[1].split('|')[1])
                preTaggedTupleListThree.append(tupleItem)
        # print(preTaggedTupleListThree)

        # 打标签
        tagDic = OrderedDict()
        for tupleItem in preTaggedTupleListThree:
            for tagItem in tupleItem:
                tagSplitItem = tagItem.split('|')
                relation = tagSplitItem[0]
                sequence = tagSplitItem[2]
                entity = tagSplitItem[1]

                relationTagSetList.append(relation)

                enSBEIList = getEntitySBEITagList(entity)
                # print(enSBEIList)
                for enItem in enSBEIList:
                    tag = enItem + '|' + relation + '|' + sequence
                    # print(tag)
                    usedTag = enItem.split('|')[0] + '|' + relation + '|' + sequence
                    # print(usedTag)
                    tagDic[enItem.split('|')[1]] = usedTag

        # print(sentence)
        sentenceSplitList = sentence.split(' ')
        # print(sentenceSplitList)
        taggedList = []
        for word in sentenceSplitList:
            if word in tagDic.keys():
                taggedList.append(word + '/' + tagDic[word])
            else:
                taggedList.append(word + '/' + 'O')
        if taggedList:
            print(taggedList)
            print(j)
            j += 1

            outputList.append(' '.join(taggedList))

    print('outputList length:',len(outputList))
    writeList2Txt(rootPath + 'sentence_tags.txt',outputList)

    # relationTagSetList = list(set(relationTagSetList))
    # print('关系标签长度：',len(relationTagSetList))
    # writeList2Txt(rootPath + 'relation_tags.txt',relationTagSetList)




















