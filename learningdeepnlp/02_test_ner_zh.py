#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode
import pickle
import codecs
import deepnlp
deepnlp.download('ner')  # download the NER pretrained models from github if installed from pip

from deepnlp import segmenter
from deepnlp import ner_tagger
# tagger = ner_tagger.load_model(lang = 'zh')

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print (" ".join(words))

print('展示words的类型：')
print(type(words),words)

# 这里添加对象持久化代码
fw = codecs.open('./words.pickle','wb')
pickle.dump(words,fw)
fw.close()

#tagger = ner_tagger.load_model(lang='zh')

#NER tagging
#tagging = tagger.predict(words)
#for (w,t) in tagging:
    #str = w + "/" + t
    #print (str.encode('utf-8'))

#Results
#我/nt
#爱/nt
#吃/nt
#北京/p
#烤鸭/nt

