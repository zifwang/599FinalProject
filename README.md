# 599FinalProject
Estimation of Origin to Destination Matrices using Link Flow Measured Data from Transportation Network
# Install Dependencies:
    Make sure you have Python3.6
    pip3 install -r requirements.txt
# Code List:
    1. data_preparation.py: 
    File Purpose: This file is used to Getting Data from given directories for the final project
    Two main functions: 
            1. load_flow_directory_files(directory): load flows from a given directory which contains flow data. Example, load_flow_directory_files('../data/TrainingFlow')
            2. load_OD_directiory_files(directory,flows_file_sequence,flows): load OD matrix from a top folder contains OD matrix data. For example, in this project, OD matrix (ls_od_1_0.csv) is in OD_1 folder which is in TrainingOD folder. So, the input of this function is the path of TrainingOD. Example, load_OD_directiory_files('../data/TrainingOD',sequence from load_flow_directory_files, flows from load_flow_directory_files)
    
    2. util.py: 
    File Purpose: implements some utility functions for training our RNN
    Functions: See the comments inside of each function for details.
        1. flow_data_pca(): used to reduce the training input dataset's size.
        2. OD_data_pca(): used to reduce the training output (ground truth) dataset's size.
        3. training_data_generation(): use to generate shuffled training data.

    3. rnn_training_cpu.py:
    File Purpose: train rnn
    
    4. rnn_training_gpu.py:
    File Purpose: gpu version of training

    Codes above have in the same directory

# Execute the program:
    python3 rnn_training_cpu.py or rnn_training_gpu.py