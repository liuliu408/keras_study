#coding:utf-8
import os

for file in os.listdir('train_img'):
    if file=='.DS_Store':
        print (file)
        print (os.remove('train_img/'+file))



