#coding:utf-8
 #*****Face Trainning*****
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D,Dense,Activation
from keras.optimizers import Adam
from keras.utils import np_utils

#Pre process images
class PreFile(object):
    def __init__(self,SrcPath,Train_folder):
        self.FilePath = SrcPath
        self.Train_folder = Train_folder

    def remove_train_folder(self):
        for i in os.listdir(self.Train_folder):
            os.remove(self.Train_folder+'/'+i)

    def FileResize(self,Width,Height):
        self.remove_train_folder()
        for type in os.listdir(self.FilePath):  # 获取文件下每个类别
            if type == '.DS_Store':
                os.remove(self.FilePath+'/'+type)
        categories = 0
        for type in os.listdir(self.FilePath): #获取文件下每个类别
            categories+=1
            folders = os.listdir(self.FilePath+'/'+type)
            for facefile in folders:
                img_open = Image.open(self.FilePath + type+'/' + facefile)
                conv_RGB = img_open.convert('L') #统一转换一下RGB格式 统一化
                new_img = conv_RGB.resize((Width,Height),Image.BILINEAR)
                new_img.save(os.path.join(self.Train_folder,os.path.basename(facefile)))
        return int(str(categories)[-1]) #转str提取下标然后转int返回

# FACE Training
class Training(object):
    def __init__(self,batch_size,number_batch,train_folder,categories):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.train_folder = train_folder
        self.categories = categories

    #Read image and return Numpy array
    def read_train_images(self,filename):
        img = Image.open(self.train_folder+filename)
        return np.array(img)

    def train(self):

        train_img_list = [] #x_train  mnist
        train_label_list = [] #y_train

        for file in os.listdir(self.train_folder):

            files_img_in_array =  self.read_train_images(filename=file)
            files_img_in_array = files_img_in_array.reshape(200,200,1)
            train_img_list.append(files_img_in_array) #Image list add up
            train_label_list.append(int(file.split('_')[0])) #lable list addup

        train_img_list = np.array(train_img_list).reshape(-1,200,200,1)
        print  train_img_list.shape
        train_label_list = np.array(train_label_list)
        print  train_label_list.shape

        train_label_list = np_utils.to_categorical(train_label_list,self.categories) #format into binary [0,0,0,0,1,0,0]

        train_img_list = train_img_list.astype('float32')
        train_img_list /= 255.0


        #-- setup Neural network CNN CNN网络层
        model = Sequential()
        #CNN Layer - 1 #input shape (100,100,1)
        model.add(Convolution2D(
            input_shape=(200,200,1),
            filters=32,#next layer output （100，100,32）
            kernel_size=(5,5), #pixel filtered
            padding='same',#外边距处理
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2,2),# Output next layer (50,50,32)
            strides=(2,2),
            padding='same'
        ))
        #CNN Layer - 2
        model.add(Convolution2D(
            filters=64,  # next layer output （50，50,64）
            kernel_size=(2, 2),  # pixel filtered
            padding='same',  # 外边距处理
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2, 2),  # Output next layer (25,25,64)
            strides=(2, 2),
            padding='same'
        ))

    #Fully connected Layer 全连接层
        model.add(Flatten()) #降维打击
        model.add(Dense(512))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.categories))
        model.add(Activation('softmax'))

    # Define Optimizer
        adam = Adam(lr=0.0001)
    #Compile the model
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
    # Fire up the network 启动网络
        model.fit(
            x=train_img_list ,
            y=train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            verbose=1,
            shuffle=True,
        )

        model.save("facemodel.h5")

def MAIN():
    #实例化文件预处理类
    PreProcess = PreFile(SrcPath='newface/',Train_folder='train_folder')
    cate = PreProcess.FileResize(Width=200,Height=200)
    Train = Training(batch_size=16,number_batch=3,train_folder='train_folder/',categories=cate)
    Train.train()

if __name__ == '__main__':
    MAIN()