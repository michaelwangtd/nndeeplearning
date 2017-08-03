# -*- coding:utf-8 -*-

import deepnlp
import jieba

deepnlp.download('ner')
# deepnlp.download()

from deepnlp import ner_tagger
tagger = ner_tagger.load_model(lang = 'zh') # Loading Chinese NER model
exit(0)
text = "习近平的妻子是彭丽媛"
words = jieba.cut(text)
print (" ".join(words).encode('utf-8'))

print('---------------------------------')

# tagging = tagger.predict(words)
# for (w,t) in tagging:
#     str = w + "/" + t
#     print (str.encode('utf-8'))
