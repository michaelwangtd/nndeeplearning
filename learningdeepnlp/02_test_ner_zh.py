#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode
import pickle
import codecs
import deepnlp
import jieba
# deepnlp.download()
# from deepnlp import segmenter
from deepnlp import ner_tagger
# tagger = ner_tagger.load_model(lang = 'zh')
tagger = ner_tagger.load_model(lang = 'en')

corpusList = []
nerList = []

# readTestTxt
# fw = codecs.open('./testSentence.txt',encoding='utf-8')
# while True:
#     line = fw.readline()
#     if line:
#         line = line.strip()
#
#         # wordList = segmenter.seg(line)
#
#         # wordList = ' '.join(jieba.cut(line)).split(' ')
#
#         wordList = line.split(' ')
#
#         print('wordList:',wordList)
#
#         taggedZip = tagger.predict(wordList)
#
#         itemReStr = ''
#         for (w,t) in taggedZip:
#             itemReStr = itemReStr + w + '/' + t + ' '
#         print('nerResult:',itemReStr)
#
#         nerList.append(itemReStr)
#     else:
#         break
#     print('nerList len:',len(corpusList))


test = 'Nicolas Sarkozy , French president , said he would consult European Union leaders on a possible boycott of the opening ceremony .'

test = test.split(' ')
print(test)
taggedZip = tagger.predict(test)
itemReStr = ''
for (w,t) in taggedZip:
    itemReStr = itemReStr + w + '/' + t + ' '
print('nerResult:',itemReStr)



#Resultss
#我/nt
#爱/nt
#吃/nt
#北京/p
#烤鸭/nt

