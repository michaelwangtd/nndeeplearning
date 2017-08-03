#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode
import pickle
import codecs
import deepnlp
import jieba
# deepnlp.download()
from deepnlp import segmenter
from deepnlp import ner_tagger
tagger = ner_tagger.load_model(lang = 'zh')

corpusList = []
nerList = []

# readTestTxt
fw = codecs.open('./testSentence.txt',encoding='utf-8')
while True:
    line = fw.readline()
    if line:
        line = line.strip()
        wordList = segmenter.seg(line)
        taggedZip = tagger.predict(wordList)
	
        itemReStr = ''
        for (w,t) in taggedZip:
            itemReStr = itemReStr + w + '/' + t + ' '
            print(itemReStr)
	    
        nerList.append(itemReStr)
    else:
        break
    print('nerList len:',len(corpusList))


#Resultss
#我/nt
#爱/nt
#吃/nt
#北京/p
#烤鸭/nt

