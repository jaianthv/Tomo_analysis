import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt
import porespy as ps
from skimage.transform import downscale_local_mean as bb


def define_header(file_coordinate,file_scalar,length,image):
    
    y_max = np.shape(image)[0]
    x_max = np.shape(image)[1]
    
    file_coordinate.write("# vtk DataFile Version 3.0\n" );
    file_coordinate.write("# vtk from Python\n" );
    file_coordinate.write("ASCII\n" );
    file_coordinate.write("DATASET STRUCTURED_POINTS\n");
    file_coordinate.write("DIMENSIONS 892 892 2\n");
    file_coordinate.write("ORIGIN 0 0 0\n");
    file_coordinate.write("SPACING 1 1 1\n")
    #file_coordinate.write("POINTS 1591328 float\n" )

    file_scalar.write("POINT_DATA 1591328\n")
    file_scalar.write("SCALARS Thickness float 1\n")
    file_scalar.write("LOOKUP_TABLE default\n")
    

def define_coordinates(file_coordinate,file_scalar,image,location):
    x_max = np.shape(image)[0]
    y_max = np.shape(image)[1]
    Sum = 0
    
    for i in range(0,x_max):
        for j in range(0,y_max):
            
            file_scalar.write("%d \n" %image[i][j])
            
            

def combine_files(file1,file2):
    filenames = [file1, file2]
    with open("output_file.vtk", "w") as outfile:
 
         for filename in filenames:

             with open(filename) as infile:

                   contents = infile.read()

                   outfile.write(contents)


folder = "H:/EEG002_X overview/reconstructed/Processed"
os.chdir(folder);

filename_coordinates=open("data_coordinates.txt","a");
filename_scalar=open("data_scalar.txt","a");
example = cv.imread("Thickness_scan_000085.tiff")
define_header(filename_coordinates,filename_scalar,1131,example)

image = "Thickness_scan_000085.tiff"
edges_calculated = cv.imread(image,-1)
define_coordinates(filename_coordinates,filename_scalar,edges_calculated,0)
  
image = "Thickness_scan_000086.tiff"
edges_calculated = cv.imread(image,-1)
define_coordinates(filename_coordinates,filename_scalar,edges_calculated,1)       
filename_coordinates.close()
filename_scalar.close()


combine_files("data_coordinates.txt","data_scalar.txt")


