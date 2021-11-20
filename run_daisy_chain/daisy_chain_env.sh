#!/bin/sh
echo "Hello there! Welcome to the Daisy-Chain segmentation environment!"
echo "To get things started, please enter the input data path. It should be a folder in:"
echo "input_path"
echo "Enter an input path below:"
read -e input_path
echo "Input Path received. Please enter the output path. It's best to input in the format:"
echo "output_path"
echo "Enter an output path below:"
read -e output_path
echo "$output_path entered."
echo "Activating the Python environment ..."
source env_path/bin/activate  # Kurt has addressed the previous problem with this environment.
# 
echo "Environment successfully activated. Running the organ daisy chain"
cd /DRIVE/pkd-data/akshay-code-2/daisy_chain  # This is the path I plan to put the daisy chain code in RadDeep, Kurt
python daisy_chain.py -i $input_path -o $output_path
echo "Daisy-Chain successfully complete"
