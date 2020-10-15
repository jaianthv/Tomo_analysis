import numpy as np
import cv2 as cv
import os
import time
import porespy as ps
import matplotlib.pyplot as plt


#step one
#get outer border

def extract_region(Reconstructed_img, minVal, maxVal, show_inbetween):
    Recon_img = cv.imread(Reconstructed_img,-1)

    '''
    cv.imshow('asd',Recon_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
   
    #correct term to adjust the threshold
    
    minimum = minVal
    maximum = maxVal
    retval, extract_region = cv.threshold(Recon_img,minVal,maxVal,cv.THRESH_BINARY)
   
    if show_inbetween == 1:
       plt.matshow(extract_region)
       plt.show()
 
    kernel = np.ones((5,5),np.uint8)
    img_filled = cv.dilate(extract_region,kernel,iterations=2)
    img_filled = np.array(img_filled, dtype=np.uint16)
    
    ret, image_filled = cv.threshold(img_filled,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)

    if show_inbetween ==1:
       plt.matshow(image_filled)
       plt.show()

    Area_coordinates = cv.findNonZero(image_filled)
    Area = len(Area_coordinates)

    #image_outer_layer_filled_thresh = np.array(image_outer_layer_filled_thresh, dtype=np.int16)
    masked_img = np.multiply(image_filled,Recon_img)

    if show_inbetween ==1:
       plt.matshow(masked_img)
       plt.show()

    '''
    cv.imshow("sds",masked_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''

    
    return masked_img, Area


    #### plotting histogram
    
def plot_histogram(masked_img):
    cv.imshow("sdf",masked_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    Z = masked_img.reshape((-1,1))
    print (Z)
    #np.hist
    x = (cv.findNonZero(masked_img))
    x = np.array(x)

    #print (len(x[1]))
    #print (x[0][0][1])
       
    Non_zero=[];
    for i in range(0,len(x)):
        coordinate = x[i]
        coordinate_x = coordinate[0][0]
        coordinate_y = coordinate[0][1]
        Non_zero.append(masked_img[coordinate_y][coordinate_x])  ### x and y are exchanged
        #print (masked_img[coordinate_y][coordinate_x])
    #print (Non_zero)
    min_non_zero = np.amin(Non_zero)
    #print (min_non_zero)
    max_non_zero = np.amax(Non_zero)

    A =[]
    for i in range(min_non_zero,max_non_zero,10):
    
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

    return min_non_zero, max_non_zero, x

    ### removing the unwated region, thresholding

    
def seperate_regions(masked_img, I_min, I_max, show_img):
    
    x = (cv.findNonZero(masked_img))
    x = np.array(x)
    #print (x)
    temp = masked_img
   
    for i in range(0,len(x)):
        coordinate_temp = x[i]
        coordinate_x = coordinate_temp[0][0]
        coordinate_y = coordinate_temp[0][1]
        if temp[coordinate_y][coordinate_x] >= I_min and temp[coordinate_y][coordinate_x] <= I_max:   #33788 - 38000 #38000 - 70000
           temp[coordinate_y][coordinate_x] = 1
        else:
           temp[coordinate_y][coordinate_x] = 0
        coordinate =[]
    if show_img == 1:
       plt.matshow(temp)
       plt.show()
    
    #### calculate the number of non-zero pixels


    #### total area is mask of 1 

  
    
    return temp, len(cv.findNonZero(temp))


def subtract_outer_layer(masked_imge,coating_image,folder_coating_layer):
    return 0

def single_image(folder, file, minVal_extract, maxVal_extract, minVal_seperate, maxVal_seperate):
    os.chdir(folder);
    #img = cv.imread(file,-1);
    img, Area = extract_region(file, minVal_extract, maxVal_extract,1)
    for_hist = img
    plot_histogram(for_hist)
    seperated, xx = seperate_regions(img, minVal_seperate, maxVal_seperate,1)
    find_thickness(seperated)
    return for_hist


def process_all(folder_recon, minVal_extract, maxVal_extract, minVal_seperate, maxVal_seperate):

    #folder_recon = "H:/Batch_1_07_2020/T_20E26_EEG001_560mg_20200910/reconstructed/"
    os.chdir(folder_recon);
    List_of_files_recon = [];
    List_of_files = os.listdir();
    
    for i in range(len(List_of_files)):
    
        Is_it_tif=".tif" in List_of_files[i]
        #print (List_of_files[i])
        if Is_it_tif==True:
           List_of_files_recon.append(List_of_files[i])

    List_of_files_recon.sort()

    #folder_save = "H:/Batch_1_07_2020/T_20E26_EEG001_560mg_20200910/reconstructed/"
    #os.chdir(folder_save)
    filename_area = open("Area_high_intensity.txt","a")
    filename_area.write("Area_whole_region,Area_Non_zero\n")
    #os.mkdir("Region_high_contrast")
    for i in range(0,len(List_of_files_recon)-600):
        
        Recon_img = folder_recon+List_of_files_recon[i]
        print (List_of_files_recon[i])
        masked_img, Area = extract_region(Recon_img,minVal_extract,maxVal_extract,0)
        seperated, len_non_zero = seperate_regions(masked_img, minVal_seperate, maxVal_seperate,0)  #12900 - 15100-lowcon 18000-20000
        thickness = find_thickness(seperated)
        os.chdir(folder_recon+"Region_high_contrast")
        #cv.imwrite("Region_high_contrast_"+List_of_files_recon[i], seperated)
        cv.imwrite("Region_high_contrast_"+List_of_files_recon[i]+"f", thickness)
        len_non_zero = len_non_zero - 2*3.14*750
        filename_area.write("%d,%d\n"%(Area,len_non_zero))
        print (len(List_of_files_recon)-i)
    filename_area.close()



def find_thickness(masked_image):
    thickness_image = ps.filters.local_thickness(masked_image)
    thickness_image = np.array(thickness_image, dtype=np.uint16)
    #plt.matshow(thickness_image)
    #plt.show()
    return thickness_image





folder = "H:/Batch_1_07_2020/EEG002_X_20200804_HECTOR/reconstructed/"
os.chdir(folder);
filename = "scan_00200.tif"



process_all(folder,18400,38700,5000,15000)
#image = single_image(folder,filename,18400,38700,5000,15000)
#plot_histogram(image)
'''
img, Recon_img = single_image()
masked_img, imag_outer_layer, Area = remove_coating_layer(img, Recon_img)
#min_val, max_val, coordinate = plot_histogram(masked_img)
seperate_regions(masked_img, 35800,38500,Area)
'''
