## Daisy Chain Utils
"""
    This will be any utility functions needed to build or test the daisy chain functionality of the end product.
"""
# Import relevant modules
from json import load
import os
from pathlib import Path
import subprocess
import numpy as np
import nibabel as nib
from numpy.core.fromnumeric import shape
# Function
# %% Combine filenames
def comb_list(name_list):
    if len(name_list) <= 3 and len(name_list) > 1:
        temp_name = ""
        for name in range(0,len(name_list)):
            temp_name += "_" + name_list[name]
        comb_name = "Combined" + temp_name
    elif len(name_list) > 3:
        comb_name = "Combined"
    return comb_name
# OS paths
# %% I think this is  a silly warning coming form the code. I can try testing this in RadDeep and debugging there if worst comes to worst
def path_slash(name):
    
    """
        There is no input here, you just use this to output the following folder
        identifiers based on the operating system.
        Windows: backslash             
        Unix: "/"
    """

    if name == 'nt':
        return "\\"
    elif name == 'posix':
        return "/"


# %% Function: list_folders. Original Author: Jinwei Zhang
def list_folders(rootDir = '.', sort=0):

    if not sort:
        return [os.path.join(rootDir, filename)
            for filename in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, filename))]
    else:
        return [os.path.join(rootDir, filename) for filename in sorted(os.listdir(rootDir))]


# %% Get the folder from a file
def get_folder(root_dir = '.'):

    idx = -1
    char_tmp = root_dir[idx]  # The end of the string
    folder = ""  # Empty initialized string
    while char_tmp != path_slash(os.name):
        folder += char_tmp  # Building the MRN backwards
        idx -= 1  # Stepping back
        char_tmp = root_dir[idx] # Next Character to evaluate in the while condition
    
    return folder[::-1]


# %% Get to the scan directories 
def get_scan_paths(root_dir = '.', child_num = 2):

    """
        For a known number of folders to the scan filders, this function will
        return the list of available scans inferred AND will return the patient
        MRN, which may be useful depending on which data I want.
        If no child number is given, then this will default an output to one child path.
    """

    temp_path = list_folders(Path(root_dir))  # Empty string that we will iteratively populate
    id = 0  # The list index of the first elemnt
    for folder in range(1,child_num):                     
        list_path = list_folders(Path(temp_path[id]))        
        temp_path = list_path
        if folder == (child_num - 3):            
            id = 0  # This will set up the next three folders
        # elif folder == (child_num - 2):
            # MRN = get_folder(temp_path[id])  # This input must be a string
        elif folder == (child_num - 1):
            scan_path = list_path
            return scan_path                     


# %% Let's just make a quick load path function
def get_load_path(load_dir, child_num = 4, scan = 0):
    list_load = get_scan_paths(load_dir, child_num)
    load_organ_scan = list_load[scan]    
    load_path = Path(load_organ_scan) / "ITKSNAP_DCM_NIFTI"
    return load_path


# %% Initialize numpy from load_path
def init_numpy_arr(load_dir, child_num = 4, scan = 0):
    load_path = get_load_path(load_dir, child_num, scan)
    load_dir = load_path / "dicom_vol.nii"    
    temp_pred_vol_nii = nib.load(load_dir)
    temp_pred = np.array(temp_pred_vol_nii.dataobj)    
    temp_init = np.zeros(np.shape(temp_pred))
    return temp_init


# %% Affine and header parameters
def pull_nifti_params(load_dir, child_num = 4, scan = 0):
    load_path = get_load_path(load_dir, child_num, scan)
    load_dir = load_path / "dicom_vol.nii"    
    temp_dicom_nii = nib.load(load_dir)
    pred_affine = temp_dicom_nii.affine
    pred_header = temp_dicom_nii.header.copy()
    return pred_affine, pred_header
    

# %% Copy path and change group
def copy_command(copy_path, destination, folder_name, group):
    path_dest = Path(destination)
    organ_folder = get_folder(copy_path)
    path_dest.mkdir(parents=True, exist_ok=True)
    print("Copying and changing group permissions...")
    rename_command = "mv " + destination + organ_folder + " " + destination + folder_name    
    copy_command = "cp -r " + copy_path + " " + destination
    chgrp_cmd = "chgrp -R " + group + " " + destination + folder_name
    full_command = copy_command + "; " + rename_command + "; " + chgrp_cmd
    subprocess.call(full_command, shell=True)        


# %% Find the midline of an np array
def find_mid_ind(img,dim):
    """
    img - Input image, or multidimensional numpy array
    dim - the dimension of the numpy array shape
    Output: the middle coordinate of the image.
    """
    img_shape = np.shape(img)
    if dim >= 0 and dim <= (len(img_shape) - 1):
        if img_shape[dim] % 2 == 0: # If the size is even
            mid_ind = 0.5*img_shape[dim] - 1 # The -1 is to account for the fact that we start at 0
        elif img_shape[dim] % 2 == 1:
            mid_ind = 0.5*(img_shape[dim] + 1) - 1 # It will move the midline index as the index of symmetry            
    else:
        print('The inserted dimension is ' + str(dim))
        print('The inserted dimension is not admittable. Plsease enter a dimension between [0,N-1]')
        mid_ind = []  # This will intentionally throw an error
    
    return np.uint(mid_ind)


# %% Paint Right Kidney Red
def kidney_repaint(mask,orig_color,new_color):
    x_mid_ind = find_mid_ind(mask,0)  # I understand this is hard coded for right now. this is the image x-axis
    N = np.shape(mask)
    x_ind = np.arange(0,N[0])
    right_half = np.int16(mask[x_ind < x_mid_ind, :, :])    
    right_half[right_half == orig_color] = new_color
    mask[x_ind < x_mid_ind, :, :] = right_half
    return np.uint16(mask)


# %% Simple mask combination
def mask_add(load_list, mask_list, child_num = 4, scan = 0): 
    temp_combine = init_numpy_arr(load_list[0], child_num, scan)   # Init from the first predictied nifti from akshay inference and the required scan so the dimensions match up    
    pred_affine, pred_header = pull_nifti_params(load_list[0], child_num, scan) # Correct scan number will have the correct header info
    for mask in range(0,len(mask_list)):                
        load_path = get_load_path(load_list[mask], child_num, scan)        
        load_dir = load_path / "pred_vol.nii"
        temp_pred_vol_nii = nib.load(load_dir)
        temp_pred = np.array(temp_pred_vol_nii.dataobj)        
        temp_combine += mask_list[mask]*temp_pred        
        # if mask == (len(mask_list) - 1):
                
    
    # temp_int = np.uint16(temp_combine)
    temp_int  = kidney_repaint(temp_combine, mask_list[0], 1) # mask_list[0] pertains to the overall kidney seg
    return temp_int
    # combined_pred_vol = nib.Nifti1Image(temp_int, affine=pred_affine, header=pred_header)         
    # nib.save(combined_pred_vol, save_dir)


# %% Retrieve the relevant diocm file
def get_dicom_vol(load_dir):

    """
        For a given load directory and file name, this will return the 
        dicom -> nifti scan, header, and affine
            mri_nifti -- dicom -> nifti scan
            nifti_header -- the copied header of the scan
            nifti_affine -- the affine of the scan. The segmentation
            and mri_nifti affines must match for things to make sense.
    """

    mri_nifti = nib.load(load_dir)
    nifti_affine = mri_nifti.affine
    nifti_header = mri_nifti.header.copy()    
    return mri_nifti, nifti_affine, nifti_header
    