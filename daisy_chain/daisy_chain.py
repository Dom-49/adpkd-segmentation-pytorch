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
print("Libraries loaded (line 24)")
# %% Preliminary data
print("Loading Preliminary Data (line 26)")
group = "pkd"
output_dir_pre = "/DRIVE/pkd-data/akshay-code-2/"  
output_dir_post = "-segmentation-pytorch"
inptut_code = " -i "
output_code = " -o "
activate_env = "env_path"
python_rel_cmd = "rel_path/inference.py"         
output_path = "output_path"

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
print("Preliminary Data Loaded (line 50)")
# %% Parser Setup
print("Setting up Arguments (line 52)...")
parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--inference_path",
    type=str,
    help="path to input dicom data (replaces path in config file)",
    default=None,
)

parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    help="path to output location",
    default=None,
)

args = parser.parse_args()

inference_path = args.inference_path
output_path = args.output_path
print("Arguments defined and loaded in")
# %% Prep the output path 
if inference_path is not None:
    inf_path = inference_path

if output_path is not None:
    out_path = output_path
    if out_path[-1] == path_slash(operating_system):
        save_base = out_path  # String that we will use to flexibly build other directories
        temp_base = Path(save_base[0:-1])
        temp_base.mkdir(parents=True, exist_ok=True)        

    elif out_path[-1] != path_slash(operating_system):
        temp_base =Path(out_path)
        temp_base.mkdir(parents=True, exist_ok=True) # Makes the MRN_Date folder so we can populate with the organ files
        save_base = out_path + path_slash(operating_system)   
        
print('Save Base Path:')
print(save_base)     
# %% Create Combined Parent Path
comb_parent_path = Path(temp_base) / comb_folder_name
comb_parent_path.mkdir(parents=True, exist_ok=True)
# %% Daisy Chain --> Runs for all organs
pred_load_dir = []
for organ in range(0,len(model_name)):
    print("Run " + str(organ + 1) + ": " + organ_name[organ] + " inference...\n")
    python_inf_dir = output_dir_pre + model_name[organ] + output_dir_post        
    cd_cmd = "cd " + python_inf_dir   
    save_path = save_base + organ_name[organ]
    # will now be a load directory for the simple combination.
    pred_load_dir.append(save_path)  # The constructed inference save directory -> load    
    run_python = python_rel_cmd + inptut_code + inf_path + output_code + save_path    
    full_command = activate_env + "; " + cd_cmd + "; " + run_python
    subprocess.call(full_command, shell=True) # This is the call to run the python code
    print(organ_name[organ] + " inference complete")          

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
