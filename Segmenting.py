import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt


#step one
#get outer border

def remove_coating_layer(image_outerlayer_thickness, Reconstructed_img):
    img = cv.imread(image_outerlayer_thickness,-1);
    Recon_img = cv.imread(Reconstructed_img,-1)

    #img = image_outerlayer_thickness
    #Recon_img = Reconstructed_img
    '''
    cv.imshow('asd',Recon_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
    #Recon_img = np.array(Recon_img,dtype=np.uint8)
    image_outer_layer = np.array(img,dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)
    image_outer_layer_dilated = cv.dilate(image_outer_layer,kernel,iterations=2)
    Invert_outer_layer = cv.bitwise_not(image_outer_layer_dilated)
    '''
    cv.imshow('asd',Invert_outer_layer)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
    #correct term to adjust the threshold
    correct_term = 1.5
    minimum = np.amin(Invert_outer_layer)+correct_term
    maximum = np.amax(Invert_outer_layer)
    retval, Invert_outer_layer = cv.threshold(Invert_outer_layer,minimum,1,cv.THRESH_BINARY)
    '''
    plt.matshow(Invert_outer_layer)
    plt.show()
    '''
    M = cv.moments(image_outer_layer_dilated)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    image_outer_layer_filled = cv.floodFill(image_outer_layer_dilated,None,(int(cx),int(cy)),255)
    image_outer_layer_filled_thresh = np.asarray(image_outer_layer_filled[1])


    minimum = np.amin(image_outer_layer_filled_thresh)
    print (minimum)
    maximum = np.amax(image_outer_layer_filled_thresh)
    print (maximum)
    retval, image_outer_layer_filled_thresh = cv.threshold(image_outer_layer_filled_thresh,minimum,1,cv.THRESH_BINARY)
    ### find area of the whole part
    '''
    plt.matshow(image_outer_layer_filled_thresh)
    plt.show()
    '''
    

    #image_outer_layer_filled_thresh = np.array(image_outer_layer_filled_thresh, dtype=np.int16)
    masked_img = np.multiply(image_outer_layer_filled_thresh,Recon_img)
    '''
    plt.matshow(masked_img)
    plt.show()
    '''
    
    masked_img = np.multiply(masked_img,Invert_outer_layer)
    masked_img = np.array(masked_img, dtype =np.uint16)
    '''
    plt.matshow(masked_img)
    plt.show()
    '''
    masked_img_binary = cv.threshold(masked_img,0,1,cv.THRESH_BINARY)
    Area = len(cv.findNonZero(image_outer_layer_filled_thresh))
    
    #plt.matshow(masked_img_binary[1])
    #plt.show()
    '''
    cv.imshow("sds",masked_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
    return masked_img, image_outer_layer, Area


    #### plotting histogram
    
def plot_histogram(masked_image):

    Z = masked_img.reshape((-1,1))
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

    
def seperate_regions(masked_img, I_min, I_max, Area):
    
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
        
    #plt.matshow(temp)
    #plt.show()
    
    #### calculate the number of non-zero pixels


    #### total area is mask of 1 

  
    
    return temp, len(cv.findNonZero(temp))


def single_image():
    folder = "H:/Batch_1_07_2020/T_20E26_EEG001_560mg_20200910/reconstructed/Processed/"
    os.chdir(folder);
    filename = "Thickness_T_20E26_EEG001_560mg_000120.tiff"
    #img = cv.imread(filename,-1);
    img = folder+filename
    folder_recon = "H:/Batch_1_07_2020/T_20E26_EEG001_560mg_20200910/reconstructed/"
    os.chdir(folder_recon);
    filename_recon = "T_20E26_EEG001_560mg_000120.tif"
    #Recon_img = cv.imread(filename_recon,-1)
    Recon_img = folder_recon+filename_recon
    return img,Recon_img


def process_all():
    folder = "H:/Batch_1_07_2020/T_20E26_EEG006_20200917/reconstructed/Processed/"
    os.chdir(folder);
    List_of_files = os.listdir();
    List_of_files_processed =[]
    for i in range(len(List_of_files)):
    
        Is_it_tif=".tiff" in List_of_files[i]
        #print (List_of_files[i])
        if Is_it_tif==True:
           List_of_files_processed.append(List_of_files[i])

    List_of_files_processed.sort()



    folder_recon = "H:/Batch_1_07_2020/T_20E26_EEG006_20200917/reconstructed//"
    os.chdir(folder_recon);
    List_of_files_recon = [];
    List_of_files = os.listdir();
    for i in range(len(List_of_files)):
    
        Is_it_tif=".tif" in List_of_files[i]
        #print (List_of_files[i])
        if Is_it_tif==True:
           List_of_files_recon.append(List_of_files[i])

    List_of_files_recon.sort()

    folder_save = "H:/Batch_1_07_2020/T_20E26_EEG006_20200917/reconstructed//"
    os.chdir(folder_save)
    filename_area = open("Area_high_intensity.txt","a")
    filename_area.write("Area_whole_region,Area_Non_zero\n")
    os.mkdir("Region_high_contrast")
    for i in range(0,len(List_of_files_processed)):
        img = folder+List_of_files_processed[i]
        Recon_img = folder_recon+List_of_files_recon[i]
        print (List_of_files_processed[i])
        masked_img, imag_outer_layer, Area = remove_coating_layer(img, Recon_img)
        seperated, len_non_zero = seperate_regions(masked_img, 38425,48500,Area)  #12900 - 15100-lowcon 18000-20000
        os.chdir(folder_save+"Region_high_contrast")
        #cv.imwrite("Region_high_contrast_"+List_of_files_processed[i], seperated)
        filename_area.write("%d,%d\n"%(Area,len_non_zero))
        print (len(List_of_files_processed)-i)
    filename_area.close()
process_all()
    
    
'''
img, Recon_img = single_image()
masked_img, imag_outer_layer, Area = remove_coating_layer(img, Recon_img)
#min_val, max_val, coordinate = plot_histogram(masked_img)
seperate_regions(masked_img, 35800,38500,Area)
'''
