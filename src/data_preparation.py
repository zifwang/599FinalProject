"""
    Author: Zifan Wang
    File Purpose: This file is used to Getting Data from given directories for the final project
    Two main functions: 
            1. load_flow_directory_files(directory): load flows from a given directory which contains flow data. Example, load_flow_directory_files('../data/TrainingFlow')
            2. load_OD_directiory_files(directory): load OD matrix from a top folder contains OD matrix data. For example, in this project, OD matrix (ls_od_1_0.csv) is in OD_1 folder which is in TrainingOD folder.
                                                    So, the input of this function is the path of TrainingOD. Example, load_OD_directiory_files('../data/TrainingOD',sequence from load_flow_directory_files, flows from load_flow_directory_files)
"""
import numpy as np
import pandas as pd
import math
import os

"""
    Methods to read flows
"""
#############################################
def load_flow_directory_files(directory):
    """
        This function is used to load all flow files from the given directory
        Arguments: directory: files location
        Returns: flows_np: a numpy array (row: samples. cols: features)
                 load_list: file number
    """
    print('-----------Getting File From %s-----------'%directory)
    position = 0
    load_list = []
    for filename in os.listdir(directory):
        # Check whether file in the directory has '.csv'. If not, skip this file
        position_csv = filename.find('.csv')
        if position_csv == -1: continue
        print('Getting File: %s'%filename)
        # Append file sequence to the load_list
        load_list.append(int(filename[filename.index('.')-1]))
        # Get flows from each file
        flow = load_flow_file(os.path.join(directory,filename))
        # In the first position, assign return flows with flow
        if position == 0:
            flows = flow
        else:
            flows = np.concatenate((flows,flow), axis = 0)
        # Increment i
        position += 1

    return flows,load_list

def load_flow_file(fileName):
    """
        This function is used to load flow data from file using pandas library
        Arguments: fileName: The .csv file name
        Returns: a numpy array: rows: samples. cols: features
    """
    # init. column names
    column_names = ['Unnamed: 0', '1', '12', '23', '25', '33', '56', '63', '65', '213', '217', '225',
                    '226', '227', '228', '230', '231', '232', '234', '238', '239', '240', '242', '243',
                    '244', '246', '248', '250', '251', '258', '259', '260', '261', '10020', '10021',
                    '10051', 'Unnamed: 36']
    
    # read from .csv file
    df = pd.read_csv(fileName, names = column_names)
    
    # If the first element in the df is np.nan, we drop this row
    if(np.isnan(df['Unnamed: 0'].iloc[0])):
        df = df.drop([0])
    
    # Check data is clean or not which means whether there is same element in 'Unnamed 0' column. If yes, remove it.
    df = data_flow_cleaning(df)

    # Drop two unnecessary rows
    df = df.drop(['Unnamed: 0','Unnamed: 36'], axis = 1)            # remove the first and the last column from the data frame
    # create a flow list
    flows = []
    for i in df.index.values:
        index_array = np.asarray(df.loc[[i]])
        flows.append(index_array.T)
    # change to numpy array
    flows = np.asarray(flows)
    samples,features,_ = flows.shape                                # get shape
    # reshape flows array
    flows = np.reshape(flows,(samples,features))                    # reshape to (samples,features)

    return flows

def data_flow_cleaning(df):
    """
        This function is used to clean flow data frame
        Arguments: df: pd dataframe
        Returns: dataFrame: a clean data frame with no repeat elements
    """
    # sort by hour
    df = df.sort_values(by=['Unnamed: 0'])
    # Check repeat and remove repeating hour
    hours = []                                                # create a list to contain hours
    # Drop all NaN
    for i in df.index.values:
        if(math.isnan(df.loc[[i]]['Unnamed: 0'])):
            df = df.drop([i])
    # Drop flows with the same hour
    for i in df.index.values:
        hour = int(df.loc[[i]]['Unnamed: 0'])
        if hour not in hours:
            hours.append(hour)
        else:
            df = df.drop([i])

    return df     
#############################################

"""
    Methods to read ODs
"""
#############################################
def load_OD_directiory_files(directory,flows_file_sequence,flows):
    """
        Function to read OD matrix from a given directory
        In order to make training X and Y be a pair, use flow files reading sequence.
        Arguments:
            1. directory: directory of OD matrix
            2. flows_file_sequence: flow files reading sequence
            3. flows: flows data loaded from a given flow directory. Use in the dimension check before return
        Return:
            1. dataMatrix: a numpy array with shape (number of samples (should be match to the number of samples in flow data),vectorized OD matrix (should be in 150 dimensions))
    """
    dictionary_OD = {}
    print('-----------Getting File From %s-----------'%directory)
    for subDirectory in os.listdir(directory):
        print('In %s'%subDirectory)
        # Skip all file does not have '.csv' suffix
        suffix = subDirectory.find('OD')
        if suffix == -1: continue
        # Find readding sequence
        index = int(subDirectory[subDirectory.index('_')+1])
        
        # Getting file from the subDirectory
        dictionary_sub_OD = {}
        for fileName in os.listdir(os.path.join(directory,subDirectory)):
            # print('Getting File: %s'%fileName)
            # Get fileID
            fileId = file_id_construction_OD(fileName)
            dictionary_sub_OD[fileId] = load_OD_file(os.path.join(os.path.join(directory,subDirectory),fileName))
        
        # Create a empty list contians vectorize OD matrix
        vectorize_OD_list = []
        # Sort the dictionary_sub_OD by the value of its keys: smaller goes first
        for i in sorted (dictionary_sub_OD):
            vectorize_OD_list.append(dictionary_sub_OD[i])

        # Update dictionary_OD matrix
        dictionary_OD[index] = np.asarray(vectorize_OD_list)
    print('Finish Getting OD File')

    # Create a list to hold the vectorized OD matrix with the same sequence as flows sequence 
    OD_list = []
    print('Flow reading sequence: ', flows_file_sequence)
    for i in flows_file_sequence:
        print('OD reading sequence: %d' % i)
        OD_list.append(dictionary_OD[i])
    OD_np = np.asarray(OD_list)
    OD_np = OD_np.reshape((OD_np.shape[0]*OD_np.shape[1],OD_np.shape[2]))

    assert(OD_np.shape[0] == flows.shape[0])
    
    return OD_np

def file_id_construction_OD(string):
    """
        Function to construction file ID in given OD matrix file
        Argument: string: fileName
        Return: fileID: a 3-4 digits number
    """
    # Find the 2nd and the 3rd underscores' positions which are first ID and sceond ID number
    position_2nd_underscore,position_3rd_underscore = find_2_3_underscore_OD_file(string)
    # Find the end position which is is the position of '.'
    end_position = string.index('.')

    # Get the first id
    firstId = ''
    for i in range (position_2nd_underscore+1,position_3rd_underscore):
        firstId = firstId + string[i]
    # Check whether firstId is only one digit. IF yes, change it to two digits
    if(int(firstId) < 10):
        firstId = '0' + firstId

    # Get the second ID
    secondId = ''
    for i in range (position_3rd_underscore+1,end_position):
        secondId = secondId + string[i]
    # Check whether secondId is only one digit. IF yes, change it to two digits
    if(int(secondId) < 10):
        secondId = '0' + secondId

    # Create fileID by combining firstId and second Id
    fileID = firstId + secondId

    return int(fileID)

def find_2_3_underscore_OD_file(string):
    """
        Function to find the second & third underscore's position in the string
        Argument: string: a fileName contains more than 3 underscore
        Return: 
            position_1: position of the second underscore
            position_2: position of the third underscore
    """
    position_1 = 0 
    position_2 = 0
    numberOfUnderscore = 0
    for i in range (0,len(string)):
        if(string[i] == '_'):
            numberOfUnderscore += 1
            if(numberOfUnderscore == 2):
                position_1 = i
            if(numberOfUnderscore == 3):
                position_2 = i
                break
    
    return position_1,position_2

def load_OD_file(fileName):
    """
        This function is used to load OD matrix data from file using pandas library
        Arguments: fileName: The .csv file name
        Returns: a numpy array: rows: samples. cols: vectorized OD matrix (should be in 149 dimensions)
    """
    # read from .csv file
    df = pd.read_csv(fileName)
    # remove the first column
    df = df.drop(['Unnamed: 0'],axis = 1)
    # Apply vectorization to the df frame
    vectorized_OD = vectorize_OD_matrix(df)

    return vectorized_OD

def vectorize_OD_matrix(df):
    """
        Function to change a pd data frame to a vector
        Arguments: df: pd data frame
        Return: non-zero entries vector of pd data frame (type np)
    """
    # Change inpute data frame to a numpy matrix
    OD_matrix = df.as_matrix()
    # Hard code define non-zero entries position for each row (a dictionary typ)
    non_zero_position = non_zero_position_reference()
    non_zero_entries = []

    for i in range (0,OD_matrix.shape[0]):
        for j in range (0,OD_matrix.shape[1]):
            non_zero_position_list = non_zero_position[i]
            if j in non_zero_position_list:
                non_zero_entries.append(OD_matrix[i,j])
    
    return np.asarray(non_zero_entries)

def non_zero_position_reference():
    """
        Function to hard code non zero entries' positions in the OD matrix
        Arguments: NO
        Returns: A python dictionary with key = row number, value = column number (in a list)
    """
    nonZeroPosition = {}
    nonZeroPosition[0] = [21,28,30]
    nonZeroPosition[1] = [21,28,30]
    nonZeroPosition[2] = [21,28,30]
    nonZeroPosition[3] = [21,28,30]
    nonZeroPosition[4] = [21,28,30]
    nonZeroPosition[5] = [21,28,30]
    nonZeroPosition[6] = [29,31]
    nonZeroPosition[7] = [29,31]
    nonZeroPosition[8] = [29,31]
    nonZeroPosition[9] = [29,31]
    nonZeroPosition[10] = [29,31]
    nonZeroPosition[11] = [21,28,30]
    nonZeroPosition[12] = [21,28,30]
    nonZeroPosition[13] = [21,28,30]
    nonZeroPosition[14] = [21,28,30]
    nonZeroPosition[15] = [21,28,30]
    nonZeroPosition[16] = [29,31]
    nonZeroPosition[17] = [29,31]    
    nonZeroPosition[18] = [21,28,30]
    nonZeroPosition[19] = [0,1,2,3,4,5,11,12,13,14,15,18]
    nonZeroPosition[20] = [0,1,2,3,4,5,6,7,8,9,10,16,17]
    nonZeroPosition[21] = []
    nonZeroPosition[22] = [0,1,2,3,4,5,11,12,13,14,15,18]
    nonZeroPosition[23] = [0,1,2,3,4,5,11,12,13,14,15,18]
    nonZeroPosition[24] = [0,1,2,3,4,5,6,7,8,9,10,16,17]
    nonZeroPosition[25] = [0,1,2,3,4,5,6,7,8,9,10,16,17]
    nonZeroPosition[26] = [0,1,2,3,4,5,11,12,13,14,15,18]
    nonZeroPosition[27] = [0,1,2,3,4,5,11,12,13,14,15,18]
    nonZeroPosition[28] = []
    nonZeroPosition[29] = []
    nonZeroPosition[30] = []
    nonZeroPosition[31] = []

    return nonZeroPosition
#############################################
# if __name__ == "__main__":
#     # Load File From Given Flow Directory
#     flows, flow_sequence = load_flow_directory_files('../data/TrainingFlow')

#     # Load File From Given OD Directory
#     vectorized_od = load_OD_directiory_files('../data/TrainingOD',flow_sequence,flows)

