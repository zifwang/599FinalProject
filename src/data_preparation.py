import numpy as np
import pandas as pd
import math
import os
""" Applying PCA function on flow vector """ 
from sklearn.decomposition import PCA 

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
        load_list.append(filename[filename.index('.')-1])
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
        

def data_flow_analysis(flows,numKernels):
    """
        This data_analysis function is used to analyze flows numpy array
        Arguments: 
            flows: input data with numpy type in the shape (number of data,features)
            numkernels: number of kernels used in PCA (constrained here make sure numKernels less than )
        Returns:
    """
    assert(numKernels <= flows.shape[1])            # Error: number of kernels are more than flows' features
    # Do PCA 
    pca = PCA(n_components = numKernels)
    pca.fit(flows)
    pca.transform(flows)

def load_OD_directory_files(directory,load_flow_sequence):
    """
        
    """
    

    


if __name__ == "__main__":
    # Load File From Given Directory
    flows, flow_sequence = load_flow_directory_files('../data/TrainingFlow')



    # # try different input days below to see pca result
    # print('print singular values:')
    # data_analysis(flows_ls,4)
