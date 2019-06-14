#coding:utf-8
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt
from PIL import Image


#加载数据集 x（60000，28，28）y(10000,)
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#数据处理-归一化 转 浮点
x_train = x_train.astype('float32')/255.  #set as type to
x_test = x_test.astype('float32')/255.
#reshape 数据形状 适用于dense层input需要
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)

#定义encoding的终极维度
encoding_dim = 2 #因为我们输出的时候 需要一个坐标 坐标 只有两个值[x=100,y=120]

#定义输入层Input可以接收的数据shape，类似tensorflow的 placeholder
input_img = Input(shape=(784,))
#定义编码层   这里是 把 数据从大维度降低到小维度 如 28*28或784 降到2个维度
#特别注意 keras 这次使用 API函数模式构建网络层
#**第一层编码**
encoded = Dense(units=128,activation='relu')(input_img)
#**第二层编码**
encoded = Dense(units=64,activation='relu')(encoded)
#**第三层编码**
encoded = Dense(units=32,activation='relu')(encoded)
#**第四层编码**--->并输出 给 解码层
encoded_output = Dense(units=encoding_dim)(encoded)

#***可以输出结果 如我想的话 2个维度**** 4 reshape(2,2)

#定义解码层
#**第一层解码**
decoded = Dense(units=32,activation='relu')(encoded_output)
#**第二层解码**
decoded = Dense(units=64,activation='relu')(decoded)
#**第三层解码**
decoded = Dense(units=128,activation='relu')(decoded)
#**第四层解码**
decoded = Dense(units=784,activation='tanh')(decoded)

#构建自动编码模型结构
autoencoder = Model(inputs=input_img,outputs=decoded)

#构建编码模型结构
encoder = Model(inputs=input_img,outputs=encoded_output)

#编译模型
autoencoder.compile(optimizer='adam',loss='mse') #均差方

#训练
autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=2,
    batch_size=512,
    shuffle=True,#每个训练epoch完成后 数据会打乱
)
autoencoder.save('xxxx.h5')
ex_img1 = Image.open('7-1.jpg')
ex_img2 = Image.open('7-2.jpg')


ex_img1 = np.array(ex_img1)
ex_img2 = np.array(ex_img2)

encoded_img1 = encoder.predict(ex_img1.reshape(1,784))
encoded_img2 = encoder.predict(ex_img2.reshape(1,784))
print (encoded_img1)
print (encoded_img2)


# 打印结果

# encoded_imgs = encoder.predict(x_test)
# print encoded_imgs[0]
# plt.scatter(x=encoded_imgs[:,0],y=encoded_imgs[:,1],c=y_test,s=3)
# plt.show()

#打印一个 三个图对比

# decoded_img = autoencoder.predict(x_test[1].reshape(1,784))
# encoded_img = encoder.predict(x_test[1].reshape(1,784))
#
# plt.figure(1)
# plt.imshow(decoded_img[0].reshape(28,28))
# plt.figure(2)
# plt.imshow(encoded_img[0].reshape(2,2))
# plt.figure(3)
# plt.imshow(x_test[1].reshape(28,28))
# plt.show()




