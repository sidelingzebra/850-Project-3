import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams["figure.dpi"] = 300


#using opencv to read an image
img0 = cv.imread("motherboard_image.jpeg")
img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) 
img_gray0=img_gray



img_gray = cv.GaussianBlur(img_gray,(7,7),3)

plt.show()

"""Threshold"""
img_gray = cv.adaptiveThreshold(img_gray, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY_INV, 5,3)



plt.imshow(img_gray)






kernel = np.ones((5,5),np.uint8)
img_gray = cv.dilate(img_gray,kernel,iterations = 8)



back=np.zeros(img_gray0.shape,np.uint8)


pts = np.array([[1458,800],[959,3473],[4821,3484],[4569,816],], np.int32)
pts = pts.reshape((-1,1,2))
focus=cv.fillPoly(back,[pts],(255,255,255))



plt.imshow(focus)



img_gray=cv.bitwise_and(img_gray,img_gray,mask=focus)
plt.imshow(img_gray)
plt.axis('off')
plt.title("Mask 1")
plt.show()

Out2=img_gray

"""Contours Detection"""
#Finding contours on the threshold image
contours, hierarchy = cv.findContours(img_gray, cv.RETR_EXTERNAL, 
                                      cv.CHAIN_APPROX_NONE)



mask=np.zeros(img_gray.shape,dtype=np.uint8)


#Ignore outside contour        
Num_cont=len(contours)


contours=cv.drawContours(mask, contours,-1, (255,255,255), 
                                  thickness=cv.FILLED)


plt.imshow(contours)
plt.axis('off')
plt.title("Contour")
plt.show()



out=cv.bitwise_and(img0,img0,mask=contours)

plt.imshow(out)
plt.axis('off')
plt.title("Final")
plt.show()

cv.imshow('Mask', Out2)
cv.imshow('out', contours)
cv.imshow('Final', out)
cv.waitKey(0)
cv.destroyAllWindows()


