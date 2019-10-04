# Import Library
import matplotlib.pyplot as plt
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import os
from fungsi import noisy
from fungsi import ftAritmatik
from fungsi import ftMedian
from fungsi import ftAlphaTrimmedMean
from fungsi import countQuality

# Open Image
# Grayscale
img = cv2.imread('img/a.jpg', cv2.IMREAD_GRAYSCALE)

# Add Noisy
img=noisy("s&p",img)
# 1. Show image
cv2.imshow('Gambar Noisy',img)
# 2. Show histogram
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr) 
plt.title("Gambar Noisy") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show() 


# A. Filter Rata rata Aritmatik
# 1. Show image
A=ftAritmatik(img)
cv2.imshow('Filter Rata rata Aritmatik',A)
print(countQuality(img,A))

# 2. Show histogram
histr1 = cv2.calcHist([A],[0],None,[256],[0,256])
plt.plot(histr1)
plt.title("Filter Rata rata Aritmatik") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show() 

# Filter Median
# 1. Show image
B=ftMedian(img)
cv2.imshow('Filter Median',B)
# 2. Show histogram
histr2 = cv2.calcHist([B],[0],None,[256],[0,256])
plt.plot(histr2)
plt.title("Filter Median") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show() 

# Filter Alpha-Trimmed Mean

# 1. Show image
C=ftAlphaTrimmedMean(img)
cv2.imshow('Filter Alpha-Trimmed Mean',C)
# 2. Show histogram
histr2 = cv2.calcHist([C],[0],None,[256],[0,256])
plt.plot(histr2)
plt.title("Filter Alpha-Trimmed Mean") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show()


