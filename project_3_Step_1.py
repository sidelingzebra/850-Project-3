import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib

#using opencv to read an image
img0 = cv.imread("motherboard_image.jpeg")

# img0=cv.resize(img0, None,fx=0.1, fy=0.1)

#Making image greyscale
img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) 

histogram,bin_edges = np.histogram(img_gray,bins=256,range=(0,256))
fig = plt.plot(histogram)
plt.show()
threshold_value = 130

img_gray = cv.GaussianBlur(img_gray,(5,5),3)
plt.imshow(img_gray)
plt.show()

"""Threshold"""

# ret,thresh = cv.threshold(blur,100,160,cv.THRESH_BINARY)

img_gray = cv.adaptiveThreshold(img_gray, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, 5,5)

# ret, img_gray = cv.threshold(img_gray,140,255,cv.THRESH_BINARY)




plt.imshow(img_gray)
plt.show()




"""Corner Detection"""
edges = cv.Canny(img_gray, 150, 300, 5)


plt.imshow(edges)
plt.show()




"""Contours Detection"""

#Finding contours on the threshold image
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, 
                                      cv.CHAIN_APPROX_NONE)



mask = np.zeros(img_gray.shape,np.uint8)

contours=cv.drawContours(mask, contours,-1, (255,255,255), thickness=-1)
# mask=np.zeros(contours.shape,dtype=np.uint8)
cv.imshow('Contours', contours)


out=cv.bitwise_and(img0,img0,mask=contours)

cv.imshow('out', out)
cv.waitKey(0)
cv.destroyAllWindows()

