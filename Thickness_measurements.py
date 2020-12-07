import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt
import porespy as ps
from skimage.transform import downscale_local_mean as bb

#Test with one image


class unwrap:
    def __init__(self,folder_name)
        self.folder_name = folder_name
        
    def openimage2array(self,filename):
        self.image = cv.imread(filename,-1)
        self.image_array = np.array(self.image,dtype=np.float32)
        return self.image_array
    
    def find_edges(self,input_array):
        
        
        
    def find_center_of_mass(self,input_array):
        self.image = input_array
        
        
        
    
    
        
        
 


def openimage(image):
    img = cv.imread(image,-1);
    #img = np.array(img,dtype=np.uint8)
    retval, dst	= cv.threshold(img,38033,55850,cv.THRESH_BINARY)
    #retval, dst	= cv.threshold(img,10,50,cv.THRESH_BINARY)

    #cv.namedWindow('image', cv.WINDOW_NORMAL);
    #cv.imshow('image',dst);
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
        
    return dst

def segment_edges(image):
    return 0

def calc_layer_thickness(image,x,y):
    #img = cv.imread(image,0)
    img =image;
    xgrad =x
    ygrad =y
    edges = cv.Canny(img,xgrad,ygrad)
    #cv.imshow('Edge_with_xgrad_%i_ygrad_%i'%(xgrad,ygrad),edges);
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #kernel = np.ones((5,5),np.uint8)
    #edges = cv.dilate(edges,kernel,iterations=1)
    lt = ps.filters.local_thickness(img)
    edges = lt
    #plt.matshow(lt)
    #plt.show()
    # add this as a input parameter in the final version
    return edges



def calc_edge(image,x,y):
    img = cv.imread(image,0)
    img =image;
    xgrad =x
    ygrad =y
    edges = cv.Canny(img,xgrad,ygrad)
    #cv.imshow('Edge_with_xgrad_%i_ygrad_%i'%(xgrad,ygrad),edges);
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #kernel = np.ones((5,5),np.uint8)
    #edges = cv.dilate(edges,kernel,iterations=1)
    #plt.matshow(lt)
    #plt.show()
    # add this as a input parameter in the final version
    return edges

def find_center_of_mass(edges_image):
    #fill the internal region and find the center of mass
    y_max = np.shape(edges_image)[0]/2
    x_max = np.shape(edges_image)[1]/2
    com_edges = edges_image.copy()
    kernel = np.ones((5,5),np.uint8)
    com_edges = cv.dilate(com_edges,kernel,iterations=1)
    
    #rectangle to find the maximum x and y radius 
    r = cv.boundingRect(com_edges)
    
    r_xmax = r[2];
    r_ymax = r[3]
    p1 = (int(r[0]),int(r[1]))
    p2 = (int((r[0])+(r[2])), int(r[1]+r[3]))
     
    cv.floodFill(com_edges,None,(int(x_max),int(y_max)),255)
    cv.imshow('imgafter filling',com_edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

    M = cv.moments(com_edges)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    #print (cx,cy)
    # x coordinate is fine +/- represents real cartisian system
    # y coordinate is not correct it is inverted + or - is reversed
    center_coordinates = (int(cx), int(cy)) 
    radius = 3
    color = (120, 120, 0) 
    thickness = 2
    image = cv.circle(com_edges, center_coordinates, radius, color, thickness)
    cv.rectangle(com_edges,p1,p2,(120,120,0),2,1)
    cv.imshow('Center_of_mass', com_edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Fill_region = ndi.binary_fill_holes(edges).astype(int)
    #edges = edges_image.astype(uint8)
    #cy, cx = ndi.center_of_mass(Fill_region)
    #plt.matshow(Fill_region)
    #plt.show()
   
    return cx,cy

def create_line(edges_image_cl,cx,cy,step):
    step = step
    y_max = np.shape(edges_image_cl)[0]
    x_max = np.shape(edges_image_cl)[1]
    print(np.shape(edges_image_cl))
    time.sleep(1)
    Thickness_all = [];
    Im = edges_image_cl.copy()
    Im_mult = edges_image_cl.copy()
    Im_for_thickness = edges_image_cl.copy()*0
    
    #first 1/2 quadrant top right
    for i in range(0,int(y_max/2),step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (x_max,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else 4 or 8 at lintype
        #cv.imshow('nm',Im)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        Im_mult = Im_mult*Im
        Im_mult = np.array(Im_mult,dtype=np.uint8)
        #print(type(Im_mult))
        #plt.matshow(Im_mult)
        #plt.show()
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        #plt.matshow(Im_mult)
        
        plt.show()
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[4])][int(temp_thickness[3])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    #plt.matshow(Im_for_thickness)
    #plt.matshow(edges_image_cl)
    #plt.show()
    #cv.imshow('nm',Im_for_thickness)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    
    #Second 1/2 quadrant bottom right
    for i in range(int(y_max/2),y_max,step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (x_max,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[4])][int(temp_thickness[3])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    plt.matshow(Im_for_thickness)
    plt.matshow(edges_image_cl)
    plt.show()
    #cv.imshow('nm',Im_for_thickness)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
     
    #third 1/2 quadrant bottom right
    for i in range(int(x_max/2),x_max,step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (i,y_max), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[4])][int(temp_thickness[3])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    plt.matshow(Im_for_thickness)
    plt.show()
    #cv.imshow('nm',Im_for_thickness)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


    #eight 1/2 quadrant top left
    for i in range(int(x_max/2),x_max,step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (i,0), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[4])][int(temp_thickness[3])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    plt.matshow(Im_for_thickness)
    plt.matshow(edges_image_cl)
    plt.show()

    
    #forth 1/2 quadrant bottom left
    for i in range(0,int(x_max/2),step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy),(i,y_max), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        #cv.waitKey(0)
        #cv.destroyAllWindows() 
        x = (cv.findNonZero(Im_mult))
        #print (x)
        #plt.matshow(Im_mult)
        #plt.show()
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[2])][int(temp_thickness[5])] = temp_thickness[0];
                                 #x                      #y
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    
    plt.matshow(Im_for_thickness)
    plt.matshow(edges_image_cl)
    plt.show()
      
    #quadrant 4 and 5 requires special attention
    #fifth 1/2 quadrant bottom left
    for i in range(int(y_max/2),int(y_max),step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (0,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        #plt.matshow(Im)
        #plt.show()
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        temp_thickness = find_distance(x)
        #may not be correct
        Im_for_thickness[int(temp_thickness[2])][int(temp_thickness[5])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    plt.matshow(Im_for_thickness)
    plt.show()
    
    #sixth 1/2 quadrant top left
    for i in range(0,int(y_max/2),step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (0,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[2])][int(temp_thickness[1])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    plt.matshow(Im_for_thickness)
    plt.show()

    
    #seventh 1/2 quadrant top left
    for i in range(0,int(x_max/2),step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (i,0), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else cv.LINE_AA 4 or 8 at lintype
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        temp_thickness = find_distance(x)
        
        Im_for_thickness[int(temp_thickness[2])][int(temp_thickness[1])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    plt.imshow(Im_for_thickness)
    plt.show()
    cv.imshow('jj',Im_for_thickness)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return Im_for_thickness, Thickness_all
   
    

















    '''
    
    for i in range(0,x_max, step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (x_max,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else 4 or 8 at lintype
        #cv.line(edges_image_cl, (cx,cy), (i,y_max), (255,100,0), thickness=cv.Line_AA, lineType=8, shift=0)
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        
        temp_thickness = find_distance(x)
        Im_for_thickness[int(temp_thickness[2])][int(temp_thickness[1])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)
    


    for i in range(0,y_max,step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (x_max,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else 4 or 8 at lintype
        #cv.line(edges_image_cl, (cx,cy), (0,i), (255,100,0), thickness=cv.Line_AA, lineType=8, shift=0)
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        
        temp_thickness = find_distance(x)
        #Im_Thickness[int(temp_thickness[2])][int(temp_thickness[1])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)

    for i in range(0,x_max-step,step):
        Im_mult = edges_image_cl.copy()
        Im = edges_image_cl.copy()
        Im = Im*0;
        cv.line(Im, (cx,cy), (x_max,i), (255,0,0), thickness=2, lineType=cv.LINE_AA, shift=0) # else 4 or 8 at lintype
        #cv.line(edges_image_cl, (cx,cy), (0,i), (255,100,0), thickness=cv.Line_AA, lineType=8, shift=0)
        Im_mult = Im_mult*Im
        x = []
        #cv.imshow('jj',Im)
        x = (cv.findNonZero(Im_mult))
        
        temp_thickness = find_distance(x)
        #Im_Thickness[int(temp_thickness[2])][int(temp_thickness[1])] = temp_thickness[0];
        Thickness_all.append(temp_thickness)
        del(temp_thickness)
        print (i)
        #cv.imshow('image',Im_mult)
        del(Im)
        del(Im_mult)

    #print ((Thickness_all[1]))
    #cv.imshow('final',Im_Thickness)
    '''

def get_edge_image(edge_image):
    return edge_image
    


        
       





        
    
    
def show_all_lines(edges_image_cl,cx,cy):
    step = 5
    y_max = np.shape(edges_image_cl)[0]
    x_max = np.shape(edges_image_cl)[1]
    for i in range(0,y_max,step):
        cv.line(edges_image_cl, (cx,cy), (x_max,i), (255,100,0), thickness=1, lineType=8, shift=0)
        #cv.imshow('imgage',edges_image_cl)

           
    for i in range(0,x_max, step):
        cv.line(edges_image_cl, (cx,cy), (i,y_max), (255,100,0), thickness=1, lineType=8, shift=0)
        #cv.imshow('imgage',edges_image_cl)
    
      
    for i in range(0,y_max,step):
        cv.line(edges_image_cl, (cx,cy), (0,i), (255,100,0), thickness=1, lineType=8, shift=0)
        #cv.imshow('imgage',edges_image_cl)
    
    
    for i in range(0,x_max-step,step):
        cv.line(edges_image_cl, (cx,cy), (i,0), (255,100,0), thickness=1, lineType=8, shift=0)
    cv.imshow('imgage',edges_image_cl)
    
    
        
        

    '''
    x_radius = x_max*cosd(angle)
    y_radius = y_max*sind(angle)
    

    '''


def find_distance(*x):
    no_of_points = len(*x)
    #print (*x)
    #print (no_of_points)
    Points = np.array(*x)
    #Im_Thickness = np.array(Points[1]).copy();
    #print ("Thickness image")
    #print ((np.shape(Im_Thickness)))
    #Im_Thickness = np.array(Im_Thickness)
    
    #Points = Points[0];
    print ("Points")
    print (len(Points))
    limits = 2
    if no_of_points == 0:
        first_inner_circle = [0,0]
        last_inner_circle = [0,0]
        Inner_points =[0,0]
        Outer_points =[0,0]
    #take first point and last point to find the reference thickness
    first_inner_circle = Points[0][0]
    
    last_inner_circle = Points[no_of_points-1][0]
    
    #Thickness = ((last_inner_circle[0] - first_inner_circle[0])**2 + (last_inner_circle[1] - first_inner_circle[1])**2)**0.5

    Inner_points =[];
    Outer_points =[];
    '''
    for i in range(0,no_of_points-1):
        if (abs(Points[i][0][0] - Points[i+1][0][0]) < limits and abs(Points[i][0][1] - Points[i+1][0][1]) < limits ):
            Inner_points.append(Points[i][0])
        else:
            Outer_points.append(Points[i][0])
        if 

            




    '''
    # i is the no of elements x[i][0][column with x and y]
    for i in range(0,no_of_points):
        if (Points[i][0][0] <= first_inner_circle[0]+limits and Points[i][0][1] <= first_inner_circle[1]+limits):
           Inner_points.append(Points[i][0])
           np.array(Inner_points)
        if (Points[i][0][0] >= first_inner_circle[0]-limits and Points[i][0][1] >= first_inner_circle[1]+limits):
           Inner_points.append(Points[i][0])
           np.array(Inner_points)

        
        else:
           Outer_points.append(Points[i][0])
           np.array(Outer_points)
    
    if len(Outer_points) == 0:
       temp =[0,0]
       Outer_points.append(temp)
    
    Inner_points_max=np.amax(Inner_points,axis=0)
    Outer_points_max=np.amax(Outer_points,axis=0)
    Inner_points_min=np.amin(Inner_points,axis=0)
    Outer_points_min=np.amin(Outer_points,axis=0)
    Inner_points = np.mean(Inner_points,axis=0)
    Outer_points = np.mean(Outer_points,axis=0)
    Inner_points = np.array(Inner_points)
    Outer_points = np.array(Outer_points)
    #Thickness = ((last_inner_circle[0] - first_inner_circle[0])**2 + (last_inner_circle[1] - first_inner_circle[1])**2)**0.5
    Thickness = ((Outer_points[0] - Inner_points[0])**2 + (Outer_points[1] - Inner_points[1])**2)**0.5
              
               
   # print(np.mean(Inner_points,axis=0))
   # print(np.mean(Outer_points,axis=0))
        #if Points[i][0][1] < last_inner_circle[1] + limits or Points[i][0][1] > last_inner_circle[1]-limits:
         #  Outer_points.append(Points[i][0])
    #print ((Inner_points))
    #print ((Outer_points))
    print (Thickness)
    
    #print (Im_Thickness[int(Outer_points[0])][int(Outer_points[1])])# + Thickness
    return Thickness, Outer_points_max[0], Outer_points_max[1], Inner_points_max[0], Inner_points_max[1],Outer_points_min[0], Outer_points_min[1], Inner_points_min[0], Inner_points_min[1]  
                           #y                    #x                   #y                    #x                     #y               #x                  #y                       #x



def define_header(file_coordinate,file_scalar,length,image):
    
    y_max = np.shape(image)[0]
    x_max = np.shape(image)[1]
    
    file_coordinate.write("# vtk DataFile Version 3.0\n" );
    file_coordinate.write("# vtk from Python\n" );
    file_coordinate.write("ASCII\n" );
    file_coordinate.write("DATASET STRUCTURED_GRID\n");
    #file_coordinate.write("DATASET UNSTRUCTURE_GRID\n")
    file_coordinate.write("DIMENSIONS %i %i %i \n" %(x_max,y_max,length));
    file_coordinate.write("POINTS %d float\n" %(x_max*y_max*length))

    file_scalar.write("POINT_DATA %d\n" %(x_max*y_max*length))
    file_scalar.write("SCALARS Thickness float 1\n")
    file_scalar.write("LOOKUP_TABLE default\n")
    

def define_coordinates(file_coordinate,file_scalar,image,location):
    x_max = np.shape(image)[0]
    y_max = np.shape(image)[1]
    Sum = 0
    
    for i in range(0,x_max):
        for j in range(0,y_max):
            if image[i][j] != 0:
               file_coordinate.write("%i %i %i \n" %(i,j,location));
               Sum = Sum+1    
               
               file_scalar.write("%d \n" %image[i][j]);
    return Sum
            

def combine_files(file1,file2):
    filenames = [file1, file2]
    with open("output_file.vtk", "w") as outfile:
 
         for filename in filenames:

             with open(filename) as infile:

                   contents = infile.read()

                   outfile.write(contents)



##############main##########################

#folder = "H:/EEG002_X overview/reconstructed/Processed"
folder = "H:/Batch_1_07_2020/T_20E26_EEG006_17092020_Hector/reconstructed"
os.chdir(folder);
List_of_files = os.listdir();
List_of_files.sort()

#filename_coordinates=open("data_coordinates.txt","a");
#filename_scalar=open("data_scalar.txt","a");
#example = openimage(List_of_files[50])
#example = bb(example,(6,6))
#define_header(filename_coordinates,filename_scalar,1131,example)

#print (List_of_files)
#List_of_tif_files=[];
#os.mkdir("Processed")
for i in range(1640,len(List_of_files)-1):
    #os.chdir(folder)
    Is_it_tif=".tif" in List_of_files[i]
    #print (List_of_files[i])
    if Is_it_tif==True:
        image = List_of_files[i]
        #image = openimage(image)
        
        #edges_calculated = cv.imread(image,-1)
        

        
        #edges_calculated = cv.normalize(edges_calculated, dst=None, alpha=0, beta=10, norm_type=cv.NORM_MINMAX)
        #edges_calculated = np.array(image, dtype=np.uint16)
        #plt.matshow(edges_calculated)
        #plt.show()
        
        #print (edges_calculated)\
        
        thresh = openimage(image)# change to thresh
        thresh = np.array(thresh,dtype=np.uint8)
        edges_calculated = calc_layer_thickness(thresh,60,60)
        
        edges_calculated = np.array(edges_calculated, dtype=np.uint16)
        os.chdir("Processed")
        cv.imwrite("Thickness_"+List_of_files[i]+"f", edges_calculated)
        #edges_calculated = bb(edges_calculated,(6,6))
        #Sum = define_coordinates(filename_coordinates,filename_scalar,edges_calculated,i)
        
        os.chdir(folder)
        print (i)
        #List_of_nxs_files.append(List_of_files[i])

        #os.system("python3 Very_simple_preprocess_NXS.py %s %s %s %s %s" %(folder,List_of_nxs_files[i],bin,Pixel_substract,step_size))


######################3main###########################
'''
folder = "H:/Batch_1_07_2020/T_20E26_EEG006_17092020_Hector/reconstructed"
os.chdir(folder);
filename = "Scan_00150.tif"
openimage(filename)
'''
'''
filename_coordinates.close()
filename_scalar.close()

filename_coordinates = open("data_coordinates.txt","r")
lines = filename_coordinates.readlines()
lines[5] = ("POINTS %d float\n"%Sum)
filename_coordinates.close()


combine_files("data_coordinates.txt","data_scalar.txt")
'''

'''
os.chdir("C:\Ghent_XTOPA")
image = "scan_000195.tif"
thresh = openimage(image)
edges_calculated = calc_layer_thickness(thresh,60,60)
#center = find_center_of_mass(edges_calculated)
#show_all_lines(edges_calculated,center[0],center[1])
#create_line(edges_calculated,center[0],center[1],1)

'''

