import matplotlib.pyplot as plt
import numpy as np
import os

#file = open("data_thickness.txt")
#lines = file.readlines()
#lines[0] = []


os.chdir('H:/Batch_1_07_2020/T_20E26_EEG006_20200917/reconstructed/')
         
with open('Area_high_intensity.txt') as f:
    lines = f.readlines()
    del(lines[0]);
    x = [line.split(",")[0] for line in lines]
    y = [line.split(",")[1] for line in lines]
    
    #y_err = [line.split(" ")[2] for line in lines]
#print (x)
x = np.array(x,dtype=np.uint32)
y = np.array(y,dtype=np.uint32)
z = np.linspace(0,len(x)-1,len(x))

print (y)
y_Area = np.divide(y,x)*100
print (max(y_Area))
sum_y_Area = sum(y_Area)
prob = y_Area//sum_y_Area
average = np.mean(y_Area)

std_dev = np.std(y_Area)
print (std_dev)
print ((np.mean(y_Area)))
plt.plot(z,y_Area,'o')
plt.xlabel("No. of slices", fontsize = 15)
plt.ylabel("High intensity area/Total area (%)", fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
text = 'Average ='+str(round(average,1))+'$\pm$' + str(round(std_dev,1))+'%'
print (text)
plt.text(10, 32, text , fontsize=15)
plt.ylim(30,75)
#plt.xscale("log")
#plt.yscale("log")
#plt.title("Average thickness per slice", fontsize = 15)
plt.show()
plt.savefig("pores_region.png")
