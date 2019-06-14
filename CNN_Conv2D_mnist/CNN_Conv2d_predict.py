#coding: utf-8
### Learnt from Mofan
### recreated by Mike G on 10th Sep 2018
### Prediction

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as processimage

#load trained model
model = load_model('liuqtrain.h5')

class MainPredictImg(object):
    def __init__(self):
        pass

    def pred(self,filename):
        #np array
        pred_img = processimage.imread(filename) #read image
        pred_img = np.array(pred_img) #tranfer to array np
        pred_img = pred_img.reshape(-1,28,28,1) #reshape into network needed shape
        prediction = model.predict(pred_img) #predict
        Final_prediction = [result.argmax() for result in prediction][0] #[1,2,3,1,3,2,2,]
        a = 0
        for i in prediction[0]:
            print (a)
            print ('Percent:{:.30%}'.format(i))
            a = a+1

        return Final_prediction

def main():
    Predict = MainPredictImg()
    res = Predict.pred('Predict_image/1.jpg')
    print ("your number is:-->",res)

if __name__ == '__main__':
    main()
