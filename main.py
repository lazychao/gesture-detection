import cv2
import numpy as np
import copy
import math
cap_region_x_begin = 0.5  # 起点/总宽度
cap_region_y_end = 0.8
isBgCaptured=False
bgSubThreshold=50
threshold = 40  # 二值化阈值
blurValue = 41  # 高斯模糊参数
learningRate = 0


def removeBg(frame):
    fgmask=bgModel.apply(frame,learningRate=learningRate)#这一步是得到前景，，注意这里提取出来的是黑白的，为什么会这样，要看提取原理

    kernel=np.ones((3,3),np.uint8)
    #3*3的卷积核
    fgmask=cv2.erode(fgmask,kernel,iterations=1)#腐蚀操作，其实就是卷积操作，目的是去除噪声
    #做到这一步是得到一个掩膜，黑白的
    #下一步进行与操作，就是和一张还没有去除背景的图进行与操作  就可以做到保留自己想要的
    #白的是1，黑的是0， 白与其他颜色相与就是不变，黑与其他相与变为黑，这样就提取出来了
    #牛逼
    cv2.imshow('mask', fgmask)
    fg=cv2.bitwise_and(frame,frame,mask=fgmask)
    return fg

def get_bgModel ():
    # camera = cv2.VideoCapture(0)
    # camera.set(10, 200)  # 设置视频属性
    # cv2.resizeWindow("trackbar", 640, 200)  # 重新设置窗口尺寸
    isBgCaptured = False
    while camera.isOpened() and isBgCaptured == False:
        cv2.waitKey(10)
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
        frame = cv2.flip(frame, 1)  # 翻转
        global bgModel
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)  # 基于自适应混合高斯背景建模的背景减除法。
        isBgCaptured = True
        print("已获取背景")



camera = cv2.VideoCapture(0)
camera.set(0, 0)   #设置视频属性
cv2.resizeWindow("trackbar", 1200, 1000)  #重新设置窗口尺寸
get_bgModel ()
while camera.isOpened():
    ret,frame=camera.read()
    #ret是布尔值  true表示获取帧正常
    frame=cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
    frame=cv2.flip(frame,1)#翻转
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)  # 画框框
    cv2.imshow('original',frame)

    fg=removeBg(frame)#得到前景了
    fg= fg[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # 剪切右上角矩形框区域
    #cv2.imshow('fg',fg)
    # 下面对前景进行处理
    # 1.转化为灰度图
    gray=cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
    #2.高斯滤波去噪
    blur=cv2.GaussianBlur(gray,(blurValue,blurValue),0)
    cv2.imshow('blur',blur)

    #3，对图像进行二值化处理
    ret,thresh=cv2.threshold(blur,threshold,255,cv2.THRESH_BINARY)
    #cv2.imshow('binary',thresh)

    #下面用特征法对图像进行判别,参数要传入二值图
    thresh1 = copy.deepcopy(thresh)  # 进行深拷贝，防止被修改
    contours,hierarchy=cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours是轮廓所组成的数组，hierarchy是轮廓属性
    length=len(contours)#轮廓数




    #获得轮廓后，找到面积最大的轮廓
    maxArea=-1
    if length>0:
        for i in range(length):
            temp=contours[i]
            area=cv2.contourArea(temp)
            if area>maxArea:
                max_i=i
                maxArea=area
        contour_biggest=contours[max_i]#记录下面积最大的轮廓,contour其实是一个点集
        hull=cv2.convexHull(contour_biggest)#得到凸包，也是一个点集
        cv2.drawContours(fg,[contour_biggest],0,(0,0,255),3)#为什么要中括号，因为要传入一个列表，绘制大轮廓
        cv2.drawContours(fg,[hull],0,(0,255,0),3)#绘制凸包
        # # 求最大区域轮廓的各阶矩，，涉及复杂的数学知识，不会。。


        moments = cv2.moments(contour_biggest)#从轮廓上（点集）算出各阶矩
        if moments['m00']!=0 :
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        cv2.circle(fg ,center, 8, (0, 0, 255), -1)  # 画出重心
        print('center是',center)
        #如何找出指尖，，有多种方法1.求重心到轮廓点的距离，极值是指尖2.求出凸缺陷，找出手指窝的个数，就可知道手指个数
        #这里采取第二种
        #那要怎么找到手指窝，可以利用函数找到所有凹陷，但不是所有凹陷都是手机窝，角度小于90度才是
        hull = cv2.convexHull(contour_biggest,returnPoints=False)
        defects=cv2.convexityDefects(contour_biggest,hull)
        #defects是一个数组，每个元素又是一个数组，分别是起始，终点，最远点，估计距离
        cnt=0
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d=defects[i][0]
                #s,e,f其实是轮廓点集的下标
                start=tuple(contour_biggest[s][0])#得到x,y坐标
                end=tuple(contour_biggest[e][0])
                far=tuple(contour_biggest[f][0])
                #求出欧氏距离
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                #余弦定理
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle<=math.pi/2:
                    cnt+=1
                    cv2.circle(fg,far,8,(0,0,255),-1)

            print('判断数字为:'+str(cnt+1))
        cv2.imshow('output',fg)

    cv2.waitKey(10)
