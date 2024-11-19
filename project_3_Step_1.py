import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib

#using opencv to read an image
img0 = cv.imread("motherboard_image.jpeg")
img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) 


"""Threshold"""
ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)


plt.imshow(thresh1)
plt.show()



