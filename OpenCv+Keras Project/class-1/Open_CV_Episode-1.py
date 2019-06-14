#coding:utf-8

import cv2
####-----静态人脸检测-------
img = cv2.imread('3.jpg')
faceModel = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
gray = cv2.cvtColor(src=img,code=cv2.COLOR_RGB2GRAY)
faces = faceModel.detectMultiScale(gray,scaleFactor=1.2)

for (x,y,w,h) in faces:
        #画外边框
        cv2.rectangle(img,pt1=(x,y), pt2=(x+w,y+h),color=(0,255,0),thickness=2)

cv2.imshow('LIVEface',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
####-----动态人脸检测-------
#导入模型
faceModel = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#开始摄像头
capture = cv2.VideoCapture(0)

while True:
    #读取摄像头数据
    ret,frame = capture.read()
    #降维打击 （28，28，3） 28，28，1
    gray = cv2.cvtColor(src=frame,code=cv2.COLOR_RGB2GRAY)
    #检测人脸 给我了一个坐标
    faces = faceModel.detectMultiScale(gray,scaleFactor=1.2)
    #标记人脸
    for (x,y,w,h) in faces:
        #画外边框
        cv2.rectangle(frame,pt1=(x,y), pt2=(x+w,y+h),color=(0,255,0),thickness=2)
        #添加名字
        cv2.putText(frame,'Mike',org=(x,y-10),fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,255,0),thickness=1)
        #显示图片
        cv2.imshow('LIVEface',frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
capture.release()
cv2.destroyAllWindows()


