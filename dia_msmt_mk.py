# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:39:43 2017

@author: mkeranen

Diameter measurement work - procedural outputs for understandings
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

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread('circles2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayblur = cv2.GaussianBlur(gray, (7, 7), 0)

#Display previous 3 operations
cv2.imshow("Original Image", image)
cv2.waitKey(0)

cv2.imshow("Converted to Grayscale", gray)
cv2.waitKey(0)

cv2.imshow("Gaussian Blur", grayblur)
cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edgedCanny = cv2.Canny(grayblur, 50, 100)
edgedDilate = cv2.dilate(edgedCanny, None, iterations=1)
edgedErode = cv2.erode(edgedDilate, None, iterations=1)
img = edgedErode.copy()
#Show image operations
cv2.imshow("Canny Edge Detection", edgedCanny)
cv2.waitKey(0)

cv2.imshow("Edge Dilation", edgedDilate)
cv2.waitKey(0)

cv2.imshow("Edge Erosion", edgedErode)
cv2.waitKey(0)

#Convert latest image back to color to allow colored contour lines
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#Find contours on grayscale image
cnts = cv2.findContours(edgedErode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]

#Draw contours on color image
img = cv2.drawContours(img.copy(), cnts, -1, (0,255,0), 2)
#Show contours
cv2.imshow("Countour Plot",img)
cv2.waitKey(0)

