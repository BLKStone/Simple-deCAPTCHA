# -*- coding:utf-8 -*-

import cv2
import numpy as np
import pytesseract
from PIL import Image
from PIL import ImageFilter
import os

class Analyzer(object):

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.imgPreprocess = None
        self.rows, self.cols, self.channels = self.img.shape
        self.height, self.width = self.img.shape[:2]
        self.m_debug = True

    # 实验的二值化方法1 大津算法
    def th1(self,img):
        # 大津算法
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    # 实验的二值化方法2 自适应阈值
    def th2(self,img):
        # 自适应阈值
        # 配合滤波
        # median = cv2.medianBlur(thresh,3)
        # img_blur = cv2.GaussianBlur(img_gray, (m_blurBlock,m_blurBlock), 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 19)
        return thresh

    # 预处理方法
    def preprocess(self):
        self.resizeCaptcha()
        # 二值化
        img_thresh = self.th1(self.img)

        if self.m_debug:
            cv2.imshow("thres", img_thresh)
            cv2.imwrite("debug/threshold.png", img_thresh)

        # 形态学运算 闭操作
        kernel = np.ones((2,2),np.uint8)
        closing = cv2.morphologyEx(img_thresh , cv2.MORPH_CLOSE, kernel)

        if self.m_debug:
            cv2.imshow("close", closing)
            cv2.imwrite("debug/closing.png",closing)

        # 扩大边界
        constant = cv2.copyMakeBorder(closing ,50,50,50,50,cv2.BORDER_CONSTANT,value=255)
        
        # 让二进制图像的 0 和 255交换
        # inverse binary image
        # constant = self.inverseColor(constant)

        # 合并
        constantSrc = cv2.merge((constant,constant,constant))

        if self.m_debug:
            cv2.imwrite("debug/broader.png",constantSrc)

        self.imgPreprocess = constantSrc.copy()

    def tessRecognize(self):

        # os.system('export TESSDATA_PREFIX="/home/user/dev/CBIR/tesseract/"')
        os.system('tesseract debug/broader.png tmp/result')

        result = ''
        try:
            with open('tmp/result.txt', 'r') as f :
                for line in f:
                    result += line
        except:
            pass

        result = result[:-2]
        
        if self.m_debug:
            print '-------------'
            print  result

        if self.m_debug:
            cv2.imshow('orignal',self.img)
            cv2.waitKey(0)

        return result


    def analyze(self):

        self.resizeCaptcha()
        # 二值化
        img_thresh = self.th1(self.img)

        if self.m_debug:
            cv2.imshow("thres", img_thresh)
            cv2.imwrite("debug/threshold.png", img_thresh)

        # 形态学运算 闭操作
        kernel = np.ones((2,2),np.uint8)
        closing = cv2.morphologyEx(img_thresh , cv2.MORPH_CLOSE, kernel)

        if self.m_debug:
            cv2.imshow("close", closing)
            cv2.imwrite("debug/closing.png",closing)

        # closing = img_thresh

        # 扩大边界
        constant = cv2.copyMakeBorder(closing ,30,30,30,30,cv2.BORDER_CONSTANT,value=255)
        
        # inverse binary image
        constant = self.inverseColor(constant)

        # http://www.bubuko.com/infodetail-1004382.html
        # 合并
        constantSrc = cv2.merge((constant,constant,constant))
        self.imgPreprocess = constantSrc.copy()

        if self.m_debug:
            cv2.imwrite('debug/constant.png',constant)
            cv2.imwrite('debug/constantSrc.png',constant)

        # 求轮廓
        # contours[0] 中存储的是每个矩形顶点的坐标，如下所示
        # [[[ 1  1]]
        #  [[ 1 38]]
        #  [[78 38]]
        #  [[78  1]]]
        constant, contours, hierarchy = cv2.findContours(constant,
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        if self.m_debug:
            for i in range(len(contours)):
                print '绘制第',i,'个轮廓'
                imgContours = cv2.drawContours(constantSrc, contours, i, (0,255,0), 2)
                cv2.imshow('contours',imgContours)
                cv2.imwrite('debug/contours.png',imgContours)
                cv2.waitKey(0)

        rotate_rects = []
        box_rects = []

        # 筛选轮廓
        for i in range(0,len(contours)):
            # mr的结构为 (top-left corner(x,y), (width, height), angle of rotation )
            mr = cv2.minAreaRect(contours[i])

            if self.verifySize(mr):
                box = cv2.boxPoints(mr)  # if you are use opencv 3.0.0
                # box = cv2.cv.boxPoints(mr) # if your are using opencv 2.4.11
                box = np.int0(box)
                rotate_rects.append(mr)
                box_rects.append(box)
            else:
                pass
        
        # 排序
        rotate_rects = sorted(rotate_rects,key = lambda x:x[0][0])  
        box_rects = sorted(box_rects,key = lambda x:x[0][0])  

        # 绘制选择后的
        if self.m_debug:
            for i in range(len(box_rects)):
                print "绘制第",i,"个矩形"
                # print box_rects[i]
                # print 'rotate rect',rotate_rects[i]
                mr =  rotate_rects[i]
                print mr
                print box_rects[i]
                # print mr[1][0]*mr[1][1]
                
                imgContoursChosen = cv2.drawContours(constantSrc, box_rects, i, (255,0,0), 1)
                cv2.imshow('contours chosen',imgContoursChosen)
                cv2.imwrite('debug/contoursChosen.png',imgContoursChosen)
                cv2.waitKey(0)

        # 提取目标
        charTarget = []   # 存储结果的list

        for i in range(0,len(rotate_rects)):

            mr = rotate_rects[i]

            # 防止出现除以 0 的错误
            if mr[1][1] == 0:
                continue

            ratio = mr[1][0] / mr[1][1]
            angle = mr[2]
            rect_size = [mr[1][0],mr[1][1]]

            print '正在处理第',i,'个矩形'

            if ( ratio > 1 ):
                angle = 90 + angle
                rect_size[0],rect_size[1] = rect_size[1],rect_size[0] # swap height and width

            # 计算矩形中心点
            center_x = (box_rects[i][0][0]+box_rects[i][1][0]+box_rects[i][2][0]+box_rects[i][3][0])/4
            center_y = (box_rects[i][0][1]+box_rects[i][1][1]+box_rects[i][2][1]+box_rects[i][3][1])/4
            center = (center_x,center_y)

            #cv2.getRotationMatrix2D(center, angle, scale) → retval
            # 获取2*3旋转矩阵
            rotmat = cv2.getRotationMatrix2D(center,angle,1)

            #cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
            imgSrc = self.imgPreprocess
            rows,cols,channels = imgSrc.shape

            rotated = cv2.warpAffine(imgSrc, rotmat,(cols,rows))
            
            print rotated .shape

            if self.m_debug:
                picname = 'debug/rotate_'+str(i)+'.png'
                cv2.imwrite(picname,rotated)

            # 接下来的目标是获取到字符碎块
            imgResized = self.showResultMat(rotated, (int(rect_size[0]),int(rect_size[1])),center , i)
            charTarget.append(imgResized)

    def verifySize(self, minAreaRect):

        # mr的结构为 (top-left corner(x,y), (width, height), angle of rotation )
        area = minAreaRect[1][0] * minAreaRect[1][1]
        if area > 6000 and area < 15000:
            return True
        return False

    def captureChar(self, img, rect_size, center, index):

        imgCorp = cv2.getRectSubPix(imgRotated,rect_size,center)

    def showResultMat(self, imgRotated, rect_size, center, index):

        m_width = 136
        m_height = 36

        imgCorp = cv2.getRectSubPix(imgRotated,rect_size,center)
        imgCorp = cv2.copyMakeBorder(imgCorp ,30,30,30,30,cv2.BORDER_CONSTANT,value=(0,0,0))
        #constant = cv2.copyMakeBorder(closing ,30,30,30,30,cv2.BORDER_CONSTANT,value=255)

        imgCorp = self.inverseColor(imgCorp)
        print 'resize',imgCorp.shape

        if self.m_debug:
            picname = 'debug/rotate_fragment_'+str(index)+'.png'
            cv2.imwrite(picname,imgCorp)

        # imgResized = cv2.resize(imgCorp,(m_width,m_height))

        # if self.m_debug:
        #     picname = 'debug/rotate_fragment_resize_'+str(index)+'.png'
        #     cv2.imwrite(picname,imgResized)

        return imgCorp

    # 反转二值图像 0 变 255, 255 变 0
    def inverseColor(self,img):

        if len(img.shape)==2:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j]==255:
                        img[i][j]=0
                    else:
                        img[i][j]=255 
        elif len(img.shape)==3:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        if img[i][j][k]==255:
                            img[i][j][k]=0
                        else:
                            img[i][j][k]=255 

        return img

    def show(self):
        cv2.namedWindow("Image")
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)

    # 缩放图像
    def resizeCaptcha(self):
        rows,cols,channels = self.img.shape
        # print rows,cols,channels 
        img_resize = cv2.resize(self.img,(cols*10,rows*10),interpolation = cv2.INTER_LINEAR)
        self.img = img_resize
        # img_thresh = self.th1(img_resize)
        # cv2.imshow("original image",self.img)
        # cv2.imshow("resize image",img_thresh)
        # cv2.waitKey(0)

    def pytessrecog(self):
        # https://bitbucket.org/3togo/python-tesseract/downloads
        image = Image.open('debug/rotate_fragment_1.png')
        print vcode

# ----------------------------------------------------------------------------------
# 以下为测试函数
def testAllpic():
        # 指明被遍历的文件夹
        rootdir = "testpic"
        right_count = 0
        result_text = ''
        
        # 遍历
        for parent,dirnames,filenames in os.walk(rootdir):
            # 输出文件信息
            for filename in filenames:

                analyzer = Analyzer(os.path.join(parent,filename))
                analyzer.preprocess()
                print filename.split('.')[0]
                result = analyzer.tessRecognize()
                result_text += filename.split('.')[0] + ' ' +result + '\n'

                if filename.split('.')[0] == result:
                    right_count += 1

            print "正确率:"+ str(float(right_count)/(len(filenames)))
            print result_text


if __name__ == '__main__':
    # analyzer = Analyzer('./testpic/4.jpg')
    # analyzer.preprocess()
    # analyzer.tessRecognize()

    testAllpic()

    # analyzer.pytessrecog()
    # analyzer.resizeCaptcha()
    # analyzer.show()


