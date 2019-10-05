# Import Library
import matplotlib.pyplot as plt
import cv2
from fungsi import noisy
from fungsi import ftAritmatik
from fungsi import ftMedian
from fungsi import ftAlphaTrimmedMean
from fungsi import countQuality

# Open Image -> Grayscale
imgAsli = cv2.imread('img/a.jpg', cv2.IMREAD_GRAYSCALE)




# Add Noisy
imgNoisy=noisy("s&p",imgAsli)
# 1. Show image
cv2.imshow('Gambar Noisy',imgNoisy)
# 2. Show histogram
histr = cv2.calcHist([imgNoisy],[0],None,[256],[0,256])
plt.plot(histr) 
plt.title("Gambar Noisy") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show() 




# A. Filter Rata rata Aritmatik
# 1. Show image
A=ftAritmatik(imgNoisy)
cv2.imshow('Filter Rata rata Aritmatik',A)

print("Mean Squared Error (Filter Rata rata Aritmetik) : ",countQuality(imgAsli,A))

# 2. Show histogram
histr1 = cv2.calcHist([A],[0],None,[256],[0,256])
plt.plot(histr1)
plt.title("Filter Rata rata Aritmatik") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show() 




# Filter Median
# 1. Show image
B=ftMedian(imgNoisy)
cv2.imshow('Filter Median',B)

print("Mean Squared Error (Filter Median) : ",countQuality(imgAsli,B))

# 2. Show histogram
histr2 = cv2.calcHist([B],[0],None,[256],[0,256])
plt.plot(histr2)
plt.title("Filter Median") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show() 




# Filter Alpha-Trimmed Mean
# 1. Show image
C=ftAlphaTrimmedMean(imgNoisy)
cv2.imshow('Filter Alpha-Trimmed Mean',C)

print("Mean Squared Error (Filter Alpha Trimmed Mean) : ",countQuality(imgAsli,C))
# 2. Show histogram
histr2 = cv2.calcHist([C],[0],None,[256],[0,256])
plt.plot(histr2)
plt.title("Filter Alpha-Trimmed Mean") 
plt.ylabel("frequency")
plt.xlabel("range")
plt.show()