#coding:utf-8
import numpy as np
from keras.models import load_model
import matplotlib.image as processimage
from PIL import Image

#load trained model
model = load_model('facemodel.h5')

class MainPredictImg(object):
    def __init__(self):
        pass

    def pred(self,filename):
        #np array
        print 'in progress'
        img_open = Image.open(filename)
        conv_RGB = img_open.convert('L')  # 统一转换一下RGB格式 统一化
        new_img = conv_RGB.resize((200, 200), Image.BILINEAR)
        new_img.save(filename)
        print 'covert done'
        pred_img = processimage.imread(filename) #read image
        pred_img = pred_img.reshape(-1, 200, 200, 1)  # reshape into network needed shape
        pred_img = pred_img.astype('float32')
        pred_img = np.array(pred_img)/255.0 #tranfer to array np

        prediction = model.predict(pred_img) #predict
        Final_prediction = [result.argmax() for result in prediction][0] #[1,2,3,1,3,2,2,]
        a = 0
        for i in prediction[0]:
            print a
            print 'Percent:{:.3%}'.format(i)
            a = a+1

        return Final_prediction

def main():
    Predict = MainPredictImg()
    res = Predict.pred('new/ph.jpg')
    print "your number is:-->",res

if __name__ == '__main__':
    main()
