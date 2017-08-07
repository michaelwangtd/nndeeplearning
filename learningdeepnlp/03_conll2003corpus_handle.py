# -*- coding:utf-8 -*-

import codecs

"""
    处理conll-2003英文语料
    语料处理成deepnlp中训练语料的格式:word/tag word2/tag2
    同时记录下所有tag标记
"""

def getSentenceList(filePath):
    resultList = []
    fr = codecs.open(filePath,encoding='utf-8')

    tempSentence = ''
    tagSetList = []

    while True:
        line = fr.readline()
        if line:
            line = line.strip()
            if ''==line:
                if tempSentence!='':
                    resultList.append(tempSentence)
                    tempSentence = ''
                    continue

            lineList = line.split(' ')
            word = lineList[0]
            tag = lineList[-1]

            tempItem = word + '/' + tag
            tempSentence = tempSentence + tempItem + ' '

            tagSetList.append(tag)
            # print(line)
        else:
            break
    tagSetList = list(set(tagSetList))
    return resultList,tagSetList


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

    # corpusPath = './corpus_train.txt'
    # taggerPath = './tagger.txt'

    sentenceList,tagSetList = getSentenceList('./eng.train')
    # sentenceList,tagSetList = getSentenceList('./test.txt')

    print(len(sentenceList))
    print(tagSetList)
    # for item in sentenceList:
    #     print(type(item),item)

    # writeList2Txt(corpusPath,sentenceList)

    # writeList2Txt(taggerPath,tagSetList)

    totalNum = len(sentenceList)
    writeList2Txt('./conll_corpus_train.txt',sentenceList[0:int(totalNum/10*6)])
    writeList2Txt('./conll_corpus_test.txt',sentenceList[int(totalNum/10*6):int(totalNum/10*8)])
    writeList2Txt('./conll_corpus_validation.txt',sentenceList[int(totalNum/10*8):])



