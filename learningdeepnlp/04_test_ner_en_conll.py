# -*- coding:utf-8 -*-

import jieba
from deepnlp import ner_tagger

if __name__ == '__main__':

    tagger = ner_tagger.load_model(lang='en')

    test = 'UK London and United States are both big city.'

    wordList = ' '.join(jieba.cut(test)).split()
    print(wordList)

    taggedZip = tagger.predict(wordList)
    itemReStr = ''
    for (w, t) in taggedZip:
        itemReStr = itemReStr + w + '/' + t + ' '
    print(itemReStr)

