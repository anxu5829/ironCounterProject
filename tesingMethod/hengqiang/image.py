# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:18:18 2018

@author: qiang heng
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Image(object):
    def __init__(self,path,form='rgb',transpose=False):
        if form=='rgb':
           self.original_image=plt.imread(path)
           self.form='rgb'
           self.grayscale_image=cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        elif form=='bgr':
           self.original_image=cv2.imread(path)
           self.form='bgr'
           self.grayscale_image=cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
           raise Exception('wrong format name')
        if transpose:
           self.original_image=self.original_image.transpose(1,0,2)
           self.grayscale_image=self.grayscale_image.transpose()
        self.rows,self.cols,_=self.original_image.shape
    
    
    def display(self,is_gray=False):
        if is_gray:
           plt.clf()
           plt.imshow(self.grayscale_image,cmap='gray')
           plt.show()
        else: 
           plt.clf()
           plt.imshow(self.original_image)
           plt.show()
           
    #low_threshold 和 high_threshold 是 cv2.canny的参数        
    def correct_direction(self,low_threshold=50,high_threshold=150):
        self.edges=cv2.Canny(self.original_image,low_threshold,high_threshold)
        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, 300)
        thetas = lines.reshape(-1, 2)[:, 1]
        theta_high = np.percentile(thetas, 85)
        theta_low = np.percentile(thetas, 15)
        theta_Mean = thetas[(thetas <= theta_high) * (thetas >= theta_low)].mean()  
        rotation_matrix=cv2.getRotationMatrix2D((self.cols/2,self.rows/2),theta_Mean*180/np.pi,1.1)
        self.original_image=cv2.warpAffine(self.original_image,rotation_matrix,(self.cols,self.rows))
        self.grayscale_image=cv2.warpAffine(self.grayscale_image,rotation_matrix,(self.cols,self.rows))
    
    #从指定位置裁剪图片
    def crop_image(self,row_start,row_end,col_start,col_end):
        self.original_image=self.original_image[row_start:row_end,col_start:col_end]
        self.grayscale_image=self.grayscale_image[row_start:row_end,col_start:col_end]
        self.rows,self.cols,_=self.original_image.shape
    #从图片中随机采样某几行
    def sample_line(self,line_number=1,is_gray=False,plot=True):
        if not is_gray:
            line_indexs=np.random.choice(self.rows,size=line_number,replace=False)
            lines=self.original_image[line_indexs]
            colors=['r','g','b']
            if plot:
               plt.clf()
               for i in range(3):
                   plt.plot(lines[0,:,i].flatten(),label=colors[i],color=colors[i])
               plt.legend()
               plt.show()
        else:
            line_indexs=np.random.choice(self.rows,size=line_number,replace=False)
            lines=self.grayscale_image[line_indexs]
            if plot:
               plt.clf()
               plt.plot(lines[0].flatten())
               plt.show()
            
##运行样例
'''
image1=Image('copper1.jpg')
image1.correct_direction()
image1.display()
image1.display(is_gray=True)

image2=Image('copper2.jpg',transpose=True)
image2.correct_direction()
image2.display()
image2.display(is_gray=True)
image2.crop_image(250,1500,50,900)
image2.display()
image2.sample_line()
'''