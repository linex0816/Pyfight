from collections import  deque  
import numpy as np  
import time  
#import imutils  
import cv2  
import serial
import string
import binascii
import time
import struct  
import numpy as np 

model = cv2.imread("model.jpg",0)#导入灰度图像 
img2 = cv2.imread("0.jpg",0) 

def drawMatches(img1, kp1, img2, kp2, matches): 
    modelrows = model.shape[0] 
    modelcols = model.shape[1] 
    rows2 = img2.shape[0] 
    cols2 = img2.shape[1] 

    out = np.zeros((max([modelrows,rows2]),modelcols + cols2, 3),dtype = 'uint8') 
    #拼接图像 
    out[:modelrows, :modelcols] = np.dstack([model, model,model]) 
    out[:rows2, modelcols:] = np.dstack([img2, img2,img2]) 

    for mat in matches: 
        model_idx = mat.queryIdx 
        img2_idx = mat.trainIdx 

        (modelx,modely) = modelkp[model_idx].pt 
        (x2,y2) = kp2[img2_idx].pt 
        #绘制匹配点 
        cv2.circle(out, (int(modelx),int(modely)),4,(255,255,0),1) 
        cv2.circle(out,(int(x2) + modelcols,int(y2)),4,(0,255,255),1) 
        cv2.line(out,(int(modelx),int(modely)),(int(x2) + modelcols,int(y2)),(255,0,0),1) 


    return out 



detector = cv2.ORB_create()  #启动ORB探测器

modelkp = detector.detect(model,None) #通过ORB找到图片model关键点
kp2 = detector.detect(img2,None)  #通过ORB找到图片2关键点
modelkp,modeldes = detector.compute(model,modelkp) #用ORB计算图片model描述符
kp2,des2 = detector.compute(img2,kp2) #用ORB计算图片2描述符

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True) #创建暴力matcher对象
matches = bf.match(modeldes,des2) #匹配图片model和图片2的描述符

img3 = drawMatches(model,modelkp,img2,kp2,matches[:15]) #绘制匹配上的关键点
# img3 = cv2.drawKeypoints(model,modelkp,None,color = (0,255,0),flags = 0)  #绘制关键点位置，不绘制大小和方向
 
#src_pts = np.float32([modelkp[mat.queryIdx].pt for mat in matches]).reshape(-1,1,2)
#dst_pts = np.float32([img2[mat.trainIdx].pt for mat in matches]).reshape(-1,1,2)

#M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)

#h,w =model.shape
#pts = np.float32([[0,0],[0,h - 1],[w - 1,h - 1],[w - 1,0]]).reshape(-1,1,2)
#dst = cv2.perspectiveTransform(pts,M)

#img3 = cv2.polylines(img_rgb,[np.int32(dst)],True,255,3,cv2.LINE_AA)

cv2.imwrite("orbTest.jpg",img3) 
cv2.imshow('orbTest',img3) 




cv2.waitKey(0)


