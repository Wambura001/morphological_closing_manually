import cv2
import numpy as np 
import matplotlib.pyplot as plt 

path = r"C:\Users\USER\Pictures\Screenshots\Screenshot 2025-03-15 161038.png"
image = cv2.imread(path)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshhold to get a binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# kernel 
kernel = np.ones((5, 5), np.uint8)

# dilation
dilated = cv2.dilate(binary, kernel, iterations = 1) # expand foreground objects

# erosion
closing_manual = cv2.erode(dilated, kernel, iterations = 1) # shrink expanded objects

cv2.imshow('Original Binary Image', binary)
cv2.imshow('dilated Image', dilated)
cv2.imshow('Morphological closing (Manual)', closing_manual)
cv2.waitKey(0)
