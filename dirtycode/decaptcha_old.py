# -*- coding:utf-8 -*-
import Image
import ImageEnhance
import ImageFilter
import sys

image_name = "./testpic/1.jpg"

# 去除干扰点
im = Image.open(image_name)

im = im.filter(ImageFilter.MedianFilter())
enchancer = ImageEnhance.Contrast(im)
im = im.convert('1')

im.show()

s = 12      #启始 切割点 x
t = 2       #启始 切割点 y
 
w = 10      #切割 宽 +y
h = 15      #切割 长 +x

im_new = []
for i in range(4): #验证码切割
    im1 = im.crop((s+w*i+i*2,t,s+w*(i+1)+i*2,h))
    im_new.append(im1)
 
im_new[0].show()#测试查看

xsize, ysize = im_new[0].size
gd = []
for i in range(ysize):
    tmp=[]
    for j in range(xsize):
        if( im_new[0].getpixel((j,i)) == 255 ):
            tmp.append(1)
        else:
            tmp.append(0)
    gd.append(tmp)

#看效果
for i in range(ysize):
    print gd[i]



# http://www.xuyukun.com/python-%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB/