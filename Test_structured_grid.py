import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt
import porespy as ps
from skimage.transform import downscale_local_mean as bb


def define_header(file_coordinate,file_scalar,file_cells,file_cell_types,length,image):
    
    y_max = np.shape(image)[0]
    x_max = np.shape(image)[1]
    
    file_coordinate.write("# vtk DataFile Version 3.0\n" );
    file_coordinate.write("vtk from Python\n" );
    file_coordinate.write("ASCII\n" );
    file_coordinate.write("DATASET UNSTRUCTURED_GRID\n");
    #file_coordinate.write("DIMENSIONS 892 892 2\n");
    file_coordinate.write("POINTS 1591328 float\n" )


    

    file_cell_types.write('CELL_TYPES 1\n')
    file_cell_types.write('1\n')


    file_scalar.write("POINT_DATA 1591328\n")
    file_scalar.write("SCALARS Thickness float 1\n")
    file_scalar.write("LOOKUP_TABLE default\n")
    

def define_coordinates(file_coordinate,file_scalar,image,location):
    x_max = np.shape(image)[0]
    y_max = np.shape(image)[1]
    Sum = 0
    
    for i in range(0,x_max):
        for j in range(0,y_max):
            if image[i][j] != 0:
               Sum = Sum+1;
               
               file_coordinate.write("%i %i %i\n"%(i,j,location))   
               file_scalar.write("%f \n" %image[i][j]);
    return Sum
            

def combine_files(file1,file2,file3,file4):
    filenames = [file1, file2, file3, file4]
    with open("output_file.vtk", "w") as outfile:
 
         for filename in filenames:

             with open(filename) as infile:

                   contents = infile.read()

                   outfile.write(contents)


def fill_cells(file_cells_define,total_points):
    total_int_points = total_points+1
    file_cells_define.write('CELLS 1 %d\n'%total_int_points)
    file_cells_define.write(str(total_points))
    file_cells_define.write(' ')
    
    for i in range(0,total_points):
        file_cells_define.write('%d'%i)
        file_cells_define.write(' ')
    file_cells_define.write('\n')
    


folder = "H:/EEG002_X overview/reconstructed/Processed"
os.chdir(folder);

filename_coordinates=open("data_coordinates.txt","a");
filename_scalar=open("data_scalar.txt","a");
filename_cells=open("data_cells.txt","a");
filename_cell_types=open("data_cell_types.txt","a");
total_Sum = 0;

example = cv.imread("Thickness_scan_000085.tiff")
define_header(filename_coordinates,filename_scalar,filename_cells,filename_cell_types,1131,example)

image = "Thickness_scan_000085.tiff"
edges_calculated = cv.imread(image,-1)
temp = define_coordinates(filename_coordinates,filename_scalar,edges_calculated,0) 
total_Sum = total_Sum + temp

image = "Thickness_scan_000086.tiff"
edges_calculated = cv.imread(image,-1)
temp = define_coordinates(filename_coordinates,filename_scalar,edges_calculated,1)
total_Sum = total_Sum + temp
fill_cells(filename_cells,total_Sum)
print (total_Sum)


filename_coordinates.close()
filename_scalar.close()
filename_cells.close()
filename_cell_types.close()


combine_files("data_coordinates.txt","data_cells.txt","data_cell_types.txt","data_scalar.txt")


