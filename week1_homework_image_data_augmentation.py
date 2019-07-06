# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:12:39 2019

@author: dell
"""

import cv2
import random
import numpy as np

# perspective transform的函数
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

#change color函数
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


#对图片实现多种变换并保存
def image_data_aug(img):
        
    # image_crop
    img_crop = img[int(img.shape[0]/4):int(3*img.shape[0]/4),0:int(3*img.shape[1]/4)]#根据图像大小选择参数大小
   
    # change_color
    img_random_color = random_light_color(img_crop)
    
    # rotation
    M = cv2.getRotationMatrix2D((img_random_color.shape[1] / 2, img_random_color.shape[0] / 2), 30, 1) # center, angle, scale
    img_rotate = cv2.warpAffine(img_random_color, M, (img_random_color.shape[1], img_random_color.shape[0]))
    # perspective transform
    M_warp, img_warp = random_warp(img_rotate, img_rotate[0], img_rotate[1])
    
    return img_warp
    

image_address = input("请输入图片路径，(地址栏中用/代替'\\')：")#路径中不能出现中文字符
img_transform = input("选择图片保存路径：")#保存在同一文件夹下，只用输入文件命名，注意加拓展名，文件名不能出现中文字符

img = cv2.imread(image_address)
cv2.imshow('image_data',image_data_aug(img))
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite(img_transform,image_data_aug(img))#保存图片


