#coding:utf-8
import cv2
import random
from PIL import Image
from time import sleep
import os
import training
####-----动态人脸检测-------
#判断目录
def find_dir(Person_name):
    #读取目录
    findfile = os.listdir("newface")
    # 转列表 然后从新排序
    list = []
    for i in findfile:
        list.append(int(i.split('_')[0]))
    list.sort() #大小排序
    #判断目录是否为空 ，空写入0 否则按照顺序添加
    if list == []:
        os.mkdir('newface/0_%s'%Person_name)
        folder_number = '0'
    else:
        #[0,1]
        folder_number = str(int(list[-1]) + 1)
        os.mkdir('newface/%s_%s' %(folder_number,Person_name) )
    return folder_number

#导入模型
def collect_face():
    faceModel = cv2.CascadeClassifier('../OpenCv_Bili_Class_1/haarcascade_frontalface_alt.xml')
    #开始摄像头
    capture = cv2.VideoCapture(0)
    PersonName=raw_input('请输入姓名:-->')
    FolderName = find_dir(PersonName)
    counter=0
    Number_samples = 200
    sleep(2)
    while True:
        counter+=1
        print counter
        #读取摄像头数据
        ret,frame = capture.read()
        #降维打击 （28，28，3） 28，28，1
        gray = cv2.cvtColor(src=frame,code=cv2.COLOR_RGB2GRAY)
        #检测人脸 给我了一个坐标
        faces = faceModel.detectMultiScale(frame,scaleFactor=1.2,minSize=(200,200))
        #标记人脸&取图片
        for (x,y,w,h) in faces:
            #画外边框
            cv2.rectangle(frame,pt1=(x,y), pt2=(x+w,y+h),color=(0,255,0),thickness=2)
            #如果小于16张图 显示collecting
            if counter < Number_samples:
                #存图片
                random_name = str(random.randint(10000, 99999))

                final_name = 'newface/'+FolderName+'_'+PersonName+'/' +FolderName+'_'+ '%s.jpg'%str(PersonName+'_'+random_name)

                SaveImg = Image.fromarray(gray)  #PIL直接读取 数组 然后存图片
                SaveImg.save(fp=final_name)
                Img= Image.open(fp=final_name)
                CropImg = Img.crop((x-40,y-40,x+w+40,y+h+40)) #截取头像 需要适当加大截取框
                CropImg.save(fp=final_name)
                #显示图片数量
                cv2.putText(frame,'%s Images collected'%str(counter),org=(x,y-10),fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,255,0),thickness=1)
            else: #大于16 completed
                cv2.putText(frame,'NOW Training',org=(x,y-10),fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,0,255),thickness=1)

            cv2.imshow('LIVEface',frame)
        #如果已经收到足够量的样本就停
        if counter > Number_samples:
            break
        #或者x退出
        if cv2.waitKey(40) & 0xFF == ord('x'):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_face()
    #完成获取头像 立刻进行训练
    print 'Training in Progress'
    training.MAIN()





