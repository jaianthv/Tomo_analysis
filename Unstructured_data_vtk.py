import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt
import porespy as ps
from skimage.transform import downscale_local_mean as bb



def define_coordinates_scalars(file_coordinate,file_scalar,file_data_thickness,image,location,resolution):

    image = bb(image,(2,2))
    x_max = np.shape(image)[0]
    y_max = np.shape(image)[1]
    #kernel = np.ones((3,3),np.uint8)
    #image = cv.erode(image,kernel,iterations=1)
    #plt.matshow(image)
    #plt.show()
    Sum = 0
    total_int = 0
    mean = []
    std_dev = []
    
    ones = 0
    twoes = 0
    threes = 0
    fours = 0
    fives = 0
    
    for i in range(0,x_max):
        for j in range(0,y_max):
            if image[i][j] == 'nan':
                print ("it is nan")
            if image[i][j] != 0:
               Sum = Sum+1;
               
               file_coordinate.write("%i %i %i\n"%(i,j,location))   
               file_scalar.write("%f \n" %(resolution*image[i][j]))
               '''
               if image[i][j] == 1:
                   ones =ones+1
               if image[i][j] == 2:
                   twoes = twoes +1
               if image[i][j] == 3:
                   threes = threes +1
               if image[i][j] == 4:
                   fours = fours +1
               if image[i][j] == 5:
                   fives = fives+1
               '''
               total_int = total_int + image[i][j]
               std_dev.append(image[i][j])

                   
    #mean = (total_int/Sum)*resolution
    if total_int != 0:
        std_dev = std_dev*resolution
        mean = np.median(std_dev)*resolution
    #mean = np.average(std_dev)*resolution
        std_dev = np.std(std_dev)
    else:
        mean = 0
        std_dev = 0
    #mean = ((ones*1) + (twoes*2) + (threes*3) + (fours*4) +(fives*5))/(ones+twoes+threes+fours+fives)
    #mean = mean*resolution
    print ((mean,std_dev))
    file_data_thickness.write("%d %d %d\n"%(location, mean,std_dev))
    return Sum


def fill_cells(file_cells_define,total_points):
    total_int_points = total_points+1
    #file_cells_define.write('CELLS 1 %d\n'%total_int_points)
    file_cells_define.write(str(total_points))
    file_cells_define.write(' ')
    
    for i in range(0,total_points):
        file_cells_define.write('%d'%i)
        file_cells_define.write(' ')
    file_cells_define.write('\n')


def define_header(file_coordinate_header,file_scalar_header,file_cells_header,length):
    
    
    file_coordinate_header.write("# vtk DataFile Version 3.0\n" );
    file_coordinate_header.write("vtk from Python\n" );
    file_coordinate_header.write("ASCII\n" );
    file_coordinate_header.write("DATASET UNSTRUCTURED_GRID\n");
    #file_coordinate.write("DIMENSIONS 892 892 2\n");
    file_coordinate_header.write("POINTS %d float\n"%length )

    file_cells_header.write("\n")
    file_cells_header.write("CELLS 1 %d\n"%(length+1))

   
    file_scalar_header.write("POINT_DATA %d\n"%length)
    file_scalar_header.write("SCALARS High_contrast_region float 1\n")
    file_scalar_header.write("LOOKUP_TABLE default\n")







#Main script

folder = "H:/Batch_1_07_2020/T_20E26_EEG003_100mg_20200917/reconstructed/Region_high_contrast/"
os.chdir(folder);
List_of_files = os.listdir();
List_of_files.sort()

#header files
filename_coordinates_header=open("data_header_coordinates.txt","a")
filename_scalar_header=open("data_header_scalar.txt","a")
filename_cells_header=open("data_cells_header.txt","a")
#data files
filename_coordinates=open("data_coordinates.txt","a");
filename_scalar=open("data_scalar.txt","a");
filename_cells=open("data_cells.txt","a");
filename_thickness_data=open("data_thickness_data.txt","a")
filename_thickness_data.write("Slice number Average_thickness Standard_deviation\n")

                           


                           
total_Sum = 0
voxel_size = 1

#fill coordinates + scalar
for i in range(len(List_of_files)):
    #os.chdir(folder)
    Is_it_tif=".tif" in List_of_files[i]
    #print (List_of_files[i])
    if Is_it_tif==True:
        image = List_of_files[i]
        edges_calculated = cv.imread(image,-1)
        temp=define_coordinates_scalars(filename_coordinates,filename_scalar,filename_thickness_data,edges_calculated,i,voxel_size)
        total_Sum = total_Sum + temp
        #cells written later
        #Cell type already written
        #scalar header later
filename_scalar.write("\n")


#fill cells

fill_cells(filename_cells,total_Sum)

#cell types
filename_cell_types=open("data_cell_types.txt","a");
filename_cell_types.write('CELL_TYPES 1\n')
filename_cell_types.write('1\n')
filename_cell_types.write('\n')
                           

#define header

define_header(filename_coordinates_header,filename_scalar_header,filename_cells_header,total_Sum)


filename_coordinates.close()
filename_scalar.close()
filename_cells.close()
filename_cell_types.close()
filename_coordinates_header.close()
filename_scalar_header.close()
filename_cells_header.close()
filename_thickness_data.close()
        

def combine_files(file1,file2,file3,file4,file5,file6,file7):
    filenames = [file1, file2, file3, file4, file5, file6, file7]
    with open("output_file.vtk", "w") as outfile:
 
         for filename in filenames:

             with open(filename) as infile:

                   contents = infile.read()

                   outfile.write(contents)


combine_files("data_header_coordinates.txt","data_coordinates.txt","data_cells_header.txt","data_cells.txt","data_cell_types.txt","data_header_scalar.txt","data_scalar.txt")

