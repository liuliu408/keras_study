#coding:utf-8
import random
import os
import numpy as np
seed='abcdefghighklmnopqrsiuvwxyzABCDEFGHIGKLMNOKPRSTXYZ1234567890'

#第一种自定义名字办法
ran_name1 = []
for i in range(7):
    choice = random.choice(seed)
    ran_name1.append(choice)
#print '第一种方法：-》》' ,''.join(ran_name1)

#第二种自定义名字办法
ran_name2 = ''.join([name2 for name2 in random.sample(seed,6)])
# print ran_name2
ran = [i for i in range(101)]
print (np.min(ran))

#第三种自定义名字办法
ran_name3 = random.randint(100000,999999)
# print ran_name3

#获取路径中的文件名
path = '/mike/teacher/logo.jpg'
print (os.path.basename(path))

#文件名分割办法

file = 'dog_jpg_jpg_jpg'
print (file.split('_'))
