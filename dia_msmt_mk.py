# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:39:43 2017

@author: mkeranen

Diameter measurement work - need to optimize binarize, segment out main diameter
Currently finds largest contour in range and drops fits a circle in.
"""

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# load the image to process
image = cv2.imread('circles.png')

#Resize img to fit computer screen better
image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))

#Process image --> grayscale --> binarize --> Gaussian Blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binarized = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
grayblur = cv2.GaussianBlur(binarized[1], (7, 7), 0)

##Uncomment to display previous 3 operations
#cv2.imshow("Original Image", image)
#cv2.waitKey(0)
#
#cv2.imshow("Converted to Grayscale", gray)
#cv2.waitKey(0)
#

#cv2.imshow("Binarized", binarized[1])
#cv2.waitKey(0)
#
#cv2.imshow("Gaussian Blur", grayblur)
#cv2.waitKey(0)




edgedCanny2 = cv2.Canny(grayblur, 50, 100)

edgedDilate = cv2.dilate(edgedCanny2, None, iterations=1)
edgedErode = cv2.erode(edgedDilate, None, iterations=1)
img = edgedCanny2.copy()
#Show image operations
#cv2.imshow("Canny Edge Detection - blur", edgedCanny2)
#cv2.waitKey(0)


#Convert latest image back to color to allow colored contour lines
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#Find contours on grayscale image
cnts = cv2.findContours(edgedCanny2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[1]
newCnts = []
circleList = []

#Process contours by arc length and circle size
for c in cnts:
    circle = cv2.minEnclosingCircle(c)
    if cv2.arcLength(c,1)>100 and circle[1]>100:
        newCnts.append(c)
        circleList.append(circle)

#Find max radius of circle enclosing the contours
maxRadius = 0
for radius in circleList:
    if radius[1] > maxRadius:
         maxRadius = radius[1]
         maxCenter = radius[0]

#Draw the max bounding circle
img = cv2.circle(img.copy(), (int(maxCenter[0]),int(maxCenter[1])), int(maxRadius), (0,0,255), 3)
#Draw contours on color image
img = cv2.drawContours(img.copy(), newCnts, -1, (0,255,0), 1)
#Show contours
cv2.imshow("Countour Plot",img)
cv2.waitKey(0)
