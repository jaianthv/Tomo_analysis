import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt

folder = "H:/Batch_1_07_2020/EEG002_X overview/reconstructed"
os.chdir(folder);
filename = "scan_000120.tif"
masked_img = cv.imread(filename,-1);


x = (cv.findNonZero(masked_img))
x = np.array(x)
Z = masked_img.reshape((-1,1))
print (len(x[1]))
print (x[0][0][1])
       
Non_zero=[];
for i in range(0,len(x)):
    coordinate = x[i]
    coordinate_x = coordinate[0][0]
    coordinate_y = coordinate[0][1]
    Non_zero.append(masked_img[coordinate_y][coordinate_x])  ### x and y are exchanged
        #print (masked_img[coordinate_y][coordinate_x])
print (Non_zero)
min_non_zero = np.amin(Non_zero)
print (min_non_zero)
max_non_zero = np.amax(Non_zero)

A =[]
for i in range(min_non_zero,max_non_zero,256):
    
    A.append(i)
        
plt.hist(Z, bins = A) 
#plt.title("histogram")
plt.yscale("log")
plt.xscale("linear")
plt.xlabel("Intensity (a.u.)", fontsize = 15)
plt.ylabel("Counts", fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
