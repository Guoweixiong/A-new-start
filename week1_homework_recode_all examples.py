# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:37:14 2019

@author: dell
"""
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/dell/Desktop/234-1.jpg') #括号内参数改变，不为-1
cv2.imshow('exam1',img)
cv2.waitKey()#waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
cv2.destroyAllWindows()
print(img)
print(img.shape) #640 480 3  3通道？？

#image crop
img_crop = img[200:600,100:400]#根据图像大小选择参数大小
cv2.imshow('img_crop',img_crop)
cv2.waitKey()
cv2.destroyAllWindows()

#color split
B, G, R = cv2.split(img)
cv2.imshow('B',B)
cv2.imshow('G',G)
cv2.imshow('R',R)
cv2.waitKey()
cv2.destroyAllWindows()

#change color
def random_light_color(img):
    #brightness
    B,G,R = cv2.split(img)#通道拆分，顺序为BGR,不是RBG
    
    b_rand = random.randint(-50,50)#生成随机数整数n a<=n<=b
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
        
    g_rand = random.randint(-50,50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255#R[],G[],B[]都是矩阵
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
        
    r_rand = random.randint(-50,50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
        
    img_merge = cv2.merge((B,G,R)) #合并之前分离出来进行变换的通道   
    #img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return img_merge

img_random_color = random_light_color(img)
cv2.imshow('img_random_color',img_random_color)
cv2.waitKey()
cv2.destroyAllWindows()


############3
#gamma correction 调整图像灰度
img_dark = cv2.imread('C:/Users/dell/Desktop/dark.jpg')
cv2.imshow('img_dark',img_dark)
cv2.waitKey()
cv2.destroyAllWindows()

def adjust_gamma(image,gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")#数据类型转换
    return cv2.LUT(img_dark,table)#LUT函数的输入是待处理的图片，格式一定要对
img_brighter = adjust_gamma(img_dark,2)
cv2.imshow('img_dark',img_dark)
cv2.imshow('img_brighter',img_brighter)
cv2.waitKey()
cv2.destroyAllWindows()
print(img_brighter.shape)


#histogram直方图

img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])   # only for 1 channel
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(明亮度), u&v: 色度饱和度
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
cv2.waitKey()
cv2.destroyAllWindows()  


# rotation
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1) # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)

cv2.waitKey()
cv2.destroyAllWindows() 

print(M)

# set M[0][2] = M[1][2] = 0
print(M)
img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna2', img_rotate2)

cv2.waitKey()
cv2.destroyAllWindows() 
# explain translation

# scale+rotation+translation = similarity transform 缩放平移旋转
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5) # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
cv2.waitKey()
cv2.destroyAllWindows() 

print(M)

##############################
# Affine Transform 仿射变换 平行线不变，角度会变
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine image', dst)
cv2.waitKey()
cv2.destroyAllWindows() 

############################
# perspective transform 平行线和角度都会发生变化
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('image_warp', img_warp)
cv2.waitKey()
cv2.destroyAllWindows() 



