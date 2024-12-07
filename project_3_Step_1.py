import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib

#Setting Default Size
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams["figure.dpi"] = 300


#using opencv to import and rotate
img0 = cv.imread("motherboard_image.jpeg")
img0=cv.rotate(img0, cv.ROTATE_90_CLOCKWISE)
img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) 
img_gray0=img_gray


#Applying Blur
img_gray = cv.GaussianBlur(img_gray,(7,7),3)


#Thresholding
img_gray = cv.adaptiveThreshold(img_gray, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY_INV, 5,3)


#Dilating
kernel = np.ones((5,5),np.uint8)
img_gray = cv.dilate(img_gray,kernel,iterations = 8)


#Black mask
back=np.zeros(img_gray0.shape,np.uint8)

#Drawing 1st mask using coordinates
pts = np.array([[1458,800],[959,3473],[4821,3484],[4569,816],], np.int32)
pts = pts.reshape((-1,1,2))
focus=cv.fillPoly(back,[pts],(255,255,255))

#Creating 1st mask
img_gray=cv.bitwise_and(img_gray,img_gray,mask=focus)
plt.imshow(img_gray)
plt.axis('off')
plt.title("Mask 1")
plt.show()

Out2=img_gray


#Finding contours on the threshold image
contours, hierarchy = cv.findContours(img_gray, cv.RETR_EXTERNAL, 
                                      cv.CHAIN_APPROX_NONE)

#Creating 2nd mask from contours
mask=np.zeros(img_gray.shape,dtype=np.uint8)


# #Ignore outside contour        
# Num_cont=len(contours)

#Drawing Contours
contours=cv.drawContours(mask, contours,-1, (255,255,255), 
                                  thickness=cv.FILLED)

#Combining original image and mask from contours
out=cv.bitwise_and(img0,img0,mask=contours)


#Showing output
cv.imshow('Mask', Out2)
cv.imshow('out', contours)
cv.imshow('Final', out)
cv.waitKey(0)
cv.destroyAllWindows()