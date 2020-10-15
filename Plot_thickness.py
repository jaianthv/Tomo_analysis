import matplotlib.pyplot as plt
import numpy as np
import os

#file = open("data_thickness.txt")
#lines = file.readlines()
#lines[0] = []


os.chdir('H:/Batch_1_07_2020/EEG002_X_20200804_HECTOR/reconstructed/Region_high_contrast/')
         
with open('data_thickness_data.txt') as f:
    lines = f.readlines()
    del(lines[0]);
    x = [line.split(" ")[0] for line in lines]
    y = [line.split(" ")[1] for line in lines]
    #y_err = [line.split(" ")[2] for line in lines]
    
print ((x))

x= np.array(x,dtype=np.uint16)
print (x)
y= np.array(y,dtype=np.uint8)
print (y)
#y_err= np.array(y_err,dtype=np.uint16)
#print (np.mean(y_err))
average = np.mean(y)

plt.bar(x,y)
#plt.plot(x,y,'*')
plt.ylim(0,40)
#plt.xscale("log")
plt.xlabel("No. of slices", fontsize = 15)
plt.ylabel("Thickness ($\mu$m)", fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
text = 'Average ='+str(round(average,1)) + ' $\pm$ 0.9 ($\mu$m) '
plt.text(0, 32, text , fontsize=15)
#plt.yscale("linear")
#plt.xscale("log")
plt.title("Average thickness per slice", fontsize = 15)
plt.show()
