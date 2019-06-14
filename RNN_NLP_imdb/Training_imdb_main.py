#coding:utf-8
from keras.preprocessing.text import Tokenizer #建立字典
from keras.preprocessing import sequence #统一化维度
import os

def readfeed():
    feedlist = []
    #读取负面新闻
    for i in os.listdir('train/neg'):
        with open('train/neg/'+i,'r') as r:
            res = r.readline()
            feedlist.append(res)
    # 读取正面新闻
    for i in os.listdir('train/pos'):
        with open('train/pos/'+i,'r') as r:
            res = r.readline()
            feedlist.append(res)
    print len(feedlist)
    return feedlist

#建立一个2000个单词的字典（top频率出现的）
token = Tokenizer(num_words=2000)
#读取所有训练集，按照单词出现频次，构成字典
word = readfeed()
# 读取所有训练集，按单词出现的频数排序，构成字典
token.fit_on_texts(word)
#打印一共有多少个文档
print token.document_count
#打印每个单词出现的次数
# print  token.word_counts
#查看映射关系
print token.word_index
exit()
#实际单词转成数字映射关系放在列表
x_train_seq = token.texts_to_sequences(word) #[a:1 b:2 cc:3]
#这里只打印单词的映射数字
# print x_train_seq[10]


#通过pad seq功能统一每一段文字文字shape 如果大于100 从前边干掉，如果小于100 前边为0  可以设定pad方式 paddingding=pre post
x_train =  sequence.pad_sequences(x_train_seq,maxlen=200,padding='post')
all_lable = [1] * 12500 + [0] * 12500
# print x_train

#--------------------
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model=Sequential()
model.add(Embedding(
    output_dim=32,
    input_dim=2000,
    input_length=200)
         )
model.add(
    Dropout(0.1)
         )
model.add(
    SimpleRNN(units=16)
         )
model.add(
    Dense(units=256,
          activation='relu')
         )
model.add(
    Dropout(0.1)
         )

model.add(Dense(units=1,
                activation='sigmoid'
                )
          )

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x=x_train,
    y=all_lable,
    batch_size=500,
    validation_split=0.2,
    epochs=10
)
model.save('text.h5')


