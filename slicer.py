# imports
import os
import nibabel as nib
import numpy as np
import imgio
import shutil
from skimage.io import imsave
## split an mgz file into N x 256 slices where N = no. of labels
def split_files(nib_file,data_path):
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    
    # Get the numpy array
    np_file = nib_file.get_fdata().astype(np.uint16)
    
    # Get the list of all labels
    label_list = np.unique(np_file)
    
    # Create folders for all labels
    for label in label_list:
        label_path = data_path+"/label-"+str(label)
        if not os.path.isdir(label_path):
            os.mkdir(label_path)
        print ("folder %d created" %(label))
        np_file_copy=np_file
        np_file_copy=np.where(np_file_copy!=label,0,1)
        
        x,y,z=np_file_copy.shape
    
        # slice numpy array at x-axis
        for i in range(0,x-1):
            # create a slice
            img_data = np_file_copy[i,:,:]

            # store the file in the label folder
            image_name = "img" + "{:0>3}".format(str(i+1))+ ".png"
            imsave(image_name,img_data)
            output_path=data_path + "/label-"+str(label)
            shutil.move(image_name,output_path)
            
    return label_list
    
    # Load the dataset
data_path="/home/sandip/Demo/4_samples"

# create output folder
output_path = data_path + "/output"
if not os.path.isdir(output_path):
    os.mkdir(output_path)
    
    
subjects = os.listdir(data_path)
print ("#### Subjects Loaded ####")

# Iterate and split
for subject in subjects:
    # ignore .git
    if subject == ".git" or subject == "output":
        continue
        
        
    print ("## Splitting  subject",subject)
    subject_path= data_path+"/"+subject
    # get files
    files = os.listdir(subject_path)
    
    # load the segmented file
    seg_file_path = subject_path + "/aparc.a2009s+aseg.mgz"
    nib_file= nib.load(seg_file_path)
    
    ## split the files in x-axis and store in a 'label directory'
    label_list = split_files(nib_file,output_path+"/subject-"+subject)
    
print ("#### All subjects splitting completed ####")

