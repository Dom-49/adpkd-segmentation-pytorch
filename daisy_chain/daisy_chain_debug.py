# Python Daisy Chain
# Runs the daisy chain for arbitrary inferences
# Simpler iteration. Kidney-Liver-Spleen combination, softmax not yet incorporated
# By: Dom Romano
# Date: 09-14-2021
# %% Import Functions
import os
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import nibabel as nib

from matplotlib.pyplot import locator_params
from daisy_chain_utils import (
    comb_list,
    path_slash,    
    get_folder,
    get_scan_paths,
    copy_command,
    mask_add,
    get_dicom_vol,
)
# %% Preliminary data
group = "pkd"
akshay_dir_pre = "/opt/pkd-data/akshay-code-2/"  
akshay_dir_post = "-segmentation-pytorch"
inptut_code = " -i "
output_code = " -o "
activate_env = ". /opt/pkd-data/akshay-code/adpkd-segmentation-pytorch/adpkd_env_cuda_11_2/bin/activate"
python_rel_cmd = "python adpkd_segmentation/inference/inference.py"         
akshay_path = r"/big_data2/apkd_segmentation/storage/output/saved_inference/adpkd-segmentation-pytorch/Analysis_2/"
# akshay_path IS A HARD-CODED DESTINATION. Perhaps smarter iterations will ask for a save path.                   
# Model Config. DO NOT MODIFY UNLESS A NEW MODEL IS ALREADY TRAINED
operating_system = os.name
organ_name = ("kidney", "liver", "spleen")
model_name = ("adpkd", "liver", "spleen")
organ_color = (2,4,8)  # ITK-SNAP label colors
# The organ color for the kidney is set to green because there seems to be a bug in the loop.
# Whenever I set the color constant to a certain number, there is an issue in the loop and color map.
# I will try to debug, but it appears that the mistake is more or less subtle.
scan_child_num = 4
youngest_child = "ITKSNAP_DCM_NIFTI"  # Last folder before dicom_vol and pred_vol nifti
dicom_vol = "dicom_vol.nii"
pred_vol = "pred_vol.nii"
comb_folder_name = comb_list(organ_name)
combined_pred_filename = "comb_pred_vol.nii"
# %% Parser Setup
parser = ArgumentParser()
parser.add_argument(
    "-t",
    "--target_path",
    type=str,
    help="path to combine the inferences",
    default=None,
)

args = parser.parse_args()

target_path = args.target_path

if target_path is not None:
    tar_path = target_path
    if tar_path[-1] == path_slash(operating_system):
        save_base = tar_path  # String that we will use to flexibly build other directories
        temp_base = Path(save_base[0:-1])
        temp_base.mkdir(parents=True, exist_ok=True)        

    elif tar_path[-1] != path_slash(operating_system):
        temp_base =Path(tar_path)
        temp_base.mkdir(parents=True, exist_ok=True) # Makes the MRN_Date folder so we can populate with the organ files
        save_base = tar_path + path_slash(operating_system)
        
# %% Create the combination path
comb_parent_path = Path(temp_base) / comb_folder_name
comb_parent_path.mkdir(parents=True, exist_ok=True)
        
# %% Make the load directory
pred_load_dir = []
for organ in range(0,len(model_name)):
    target = save_base + organ_name[organ]
    pred_load_dir.append(target)  # The constructed inference save directory -> load
        
# %% Combine the Daisy Chain
print("Combining the organ segmentations...")
print("\n\n Predicted Load Dir \n\n")
print(pred_load_dir)
scan_list = get_scan_paths(pred_load_dir[0], scan_child_num)
if len(scan_list) == 1:
    print("One scan detected for this study. Processing...")
    scan_folder = get_folder(scan_list[0])
    print("Combining for " + scan_folder)
    print("Child Number: " + str(scan_child_num))
    comb_mask = mask_add(pred_load_dir, organ_color, scan_child_num, 0)
    # Constructing load path
    load_parent = Path(scan_list[0])
    load_path = load_parent / youngest_child
    print("Loading MRI NIFTI and pulling necessary parameters...")
    mri_nifti, nifti_affine, nifti_header = get_dicom_vol(load_path / dicom_vol)
    print("Saving the combined prediction mask and mri NIFTI...")
    comb_save_path = comb_parent_path / scan_folder
    comb_save_path.mkdir(parents=True, exist_ok=True)
    nib.save(mri_nifti, comb_save_path / dicom_vol)
    combined_pred_vol = nib.Nifti1Image(comb_mask, affine=nifti_affine, header=nifti_header)
    nib.save(combined_pred_vol, comb_save_path / combined_pred_filename)
    print("Files saved.")
elif len(scan_list) > 1:
    print(str(len(scan_list)) + ' scans detected.')
    for scan in range(0,len(scan_list)):
        print("SCAN: " + str(scan))
        scan_folder = get_folder(scan_list[scan]) # This will copy the scan name. Great for saving    
        print("Combining for " + scan_folder)
        comb_mask = mask_add(pred_load_dir, organ_color, scan_child_num, scan)
        # Constructing load path
        load_parent = Path(scan_list[scan])        
        load_path = load_parent / youngest_child
        print("Loading MRI NIFTI and pulling necessary parameters...")
        mri_nifti, nifti_affine, nifti_header = get_dicom_vol(load_path / dicom_vol)
        comb_save_path = comb_parent_path / scan_folder
        comb_save_path.mkdir(parents=True, exist_ok=True)
        nib.save(mri_nifti, comb_save_path / "dicom_vol.nii")
        combined_pred_vol = nib.Nifti1Image(comb_mask, affine=nifti_affine, header=nifti_header)
        nib.save(combined_pred_vol, comb_save_path / combined_pred_filename)
        if scan != (len(scan_list)-1):
            print('Combined Prediction Mask Saved. Moving to next scan...\n')
        elif scan == (len(scan_list)-1):
            print('Combined Prediction Mask Saved for all scans.\n')
# %% Change the group
print("Changing Group Permissions...")
chgrp_cmd = "chgrp -R " + group + " " + save_base
subprocess.call(chgrp_cmd, shell=True) # This changes the permisions group of the output 
print("Group changed.")
print("Processing Complete. Please check the output files in the following path:")
print(comb_parent_path)
print("Likewise, please check the copied " + organ_name[0] + " inference under: ")
print(akshay_path)
print("NOTE: The above copy path is hard-coded on line 31 of the daisy chain python script. If you wish to change the path or have control over the destination, feel free to discuss with the current repository manager.")