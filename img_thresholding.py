import cv2                          #importing all the necessary libraries
import numpy as np                                                                          
from numpy import array
low = array([0,100,80])             #initializing the lower and higher arrays for threshholding
high = array([10,256,256])          # change the values of these arrays(low, high, low1, high1) to the desired values in HSV colour space
low1 = array([170,100,80])
high1 = array([180,256,256])
area = 0.0
kernel1 = np.ones((15,15),np.uint8) #initializing kernels for morphological operations
kernel2 = np.ones((5,5),np.uint8)
frame = cv2.imread('D:\index2.jpg') #Specify the path of the input image in the brackets
cv2.imshow('image',frame)
res1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#converts the image in the RGB colour space to HSV colour space for the threshold operation
cv2.imshow('HSV_Image',res1)                #displays the image in HSV colour space
mask1 = cv2.inRange(res1,low,high)          #threshold operation
mask2 = cv2.inRange(res1,low1,high1)
maskres = mask1+mask2                       #adding the two masks
mask_result1 = cv2.bitwise_and(res1,res1,mask = mask1)# just for the purpose of understanding how the threshold operation works
mask_result2 = cv2.bitwise_and(res1,res1,mask = mask2)#''
mask_result_net = mask_result1+mask_result2
cv2.imshow('mask result net', mask_result_net)#displaying the masks
cv2.imshow('mask_result1',mask_result1)
cv2.imshow('mask_result2',mask_result2)
maskres1 = cv2.morphologyEx(maskres,cv2.MORPH_OPEN,kernel1)#morphological operations for removing unwanted noise
maskres2 = cv2.morphologyEx(maskres1,cv2.MORPH_CLOSE,kernel2)
contours, hierarchy = cv2.findContours(maskres2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#finding the contours for the threshold objects
cv2.drawContours(frame,contours,-1,(0,255,0),1)#drawing contours on the colour image
for i in range(len(contours)): #drawing bounding rectangles around all the contours
    x,y,w,h = cv2.boundingRect(contours[i])#getting the origin and the width and height of the bounding rectangle
    cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255),2)#drawing the bounding rectangle
cv2.imwrite('Final_result.jpg',frame)#writing or storing the output image in the directory
cv2.imshow('frame', frame)#displaying the output image
cv2.waitKey(0)#waiting till a key is pressed or you can specify the duration of the display in the parentheses
cv2.destroyAllWindows()#destroying all windows


