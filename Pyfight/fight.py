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
from matplotlib import pyplot as plt

#设置最小匹配值
MIN_MATCHS = 10
#载入被匹配图片和模型图片
cap = cv2.imread('pic/0.jpg',0)
model = cv2.imread('pic/model.jpg',0)

#打开ORB关键点探测器
orb = cv2.xfeatures2d.SIFT_create()

#查找关键点和描述符 
kp_model,des_model = orb.detectAndCompute(model,None)
kp_frame,des_frame = orb.detectAndCompute(cap,None)

#创建暴力matcher对象。它有两个可选参数。第一个是 normType。它是用来指定要使用的距离测试类型。
#默认值为 cv2.Norm_L2。这很适合 SIFT 和 SURF 等（c2.NORM_L1 也可以）。
#对于使用二进制描述符的 ORB，BRIEF，BRISK算法等，要使用 cv2.NORM_HAMMING，这样就返回两个测试对象之间的汉明距离。
#bf = cv2.BFMatcher()

#使用BFMatcher.knnMatch()来获得最佳匹配点，其中k=2这个值很关键
#BFMatcher 对象bf。具有两个方法，BFMatcher.match() 和 BFMatcher.knnMatch()。
#第一个方法会返回最佳匹配。第二个方法为每个关键点返回 k 个最佳匹配（降序排列之后取前 k 个），其中 k 是由用户设定的。
#matches = bf.knnMatch(des_model,des_frame,k=2)

#FLANN 是快速最近邻搜索包（Fast_Library_for_Approximate_Nearest_Neighbors）的简称。在面对大数据集时它的效果要好于 BFMatcher()
FLANN_INDEX_KDTREE = 0
#FLANN 匹配,需要传入两个字典作为参数.第一个是 IndexParams用来确定要使用的算法,对于 SIFT 和 SURF 等使用下面语句
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#对于ORB使用下面语句
#index_params= dict(algorithm = FLANN_INDEX_KDTREE,table_number = 6, key_size = 12,multi_probe_level = 1)
#第二个字典是 SearchParams用来指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多。
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
#第一个方法会返回最佳匹配。第二个方法为每个关键点返回 k 个最佳匹配（降序排列之后取前 k 个），其中 k 是由用户设定的。
matches = flann.knnMatch(des_model,des_frame,k=2)

#在matches中查找最佳点，其中每个matchs点有两个相近的最佳匹配，通过×0.75系数计算两个图片关键点的最短距离。
good = []
for m,n in matches:   #其中m是第一列元素，n是第二列元素
    if m.distance < 0.75 * n.distance:
        good.append(m)

#函数 cv2.drawMatchsKnn为每个关键点和它的 k 个最佳匹配点绘制匹配线。如果 k 等于 2，就会为每个关键点绘制两条最佳匹配直线。
#cap = cv2.drawMatchesKnn(model,kp_model,cap,kp_frame,good[:MIN_MATCHS],None,flags=2)
#print("%d个点匹配." %(len(good)))

if len(good)>MIN_MATCHS:
    # 在good里面找出每一行的模型图像中的描述符的索引m.queryIdx，通过索引值找到对应关键点的x，y值[kp_model[m.queryIdx].pt 
    #其中 这地方有一个多维数组的循环 xx for xx in array，和对得到的数组在不改变数据前提下重新整形三维数组.reshape(-1,1,2)，这里-1是一列多行，1是二维数组的1列，2是一维数组的2列
    src_pts = np.float32([ kp_model[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # 在good里面找出每一行的目标图像中的描述符的索引 m.trainIdx
    dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #这里用了cv2的RANSAC来做单应性匹配，将modle图与变形后的目标图匹配，关键算法是RANSAC。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #将多维数组降为一维数组，用于后面的画线
    matchesMask = mask.ravel().tolist()

    #绘制目标图片边框
    h,w = model.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cap = cv2.polylines(cap,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCHS))
    matchesMask = None

# Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(model,kp_model,cap,kp_frame,good,None,**draw_params)

cv2.imshow('frame',img3)
plt.imshow(img3, 'gray'),plt.show()
cv2.waitKey(0)
#销毁所有窗口  
cv2.destroyAllWindows() 