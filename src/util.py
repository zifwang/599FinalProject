"""
    Author: Zifan Wang
    Purpose: implements some utility functions for training our RNN
    Functions: See the comments inside of each function for details.
        1. flow_data_pca(): used to reduce the training input dataset's size.
        2. OD_data_pca(): used to reduce the training output (ground truth) dataset's size.
        3. training_data_generation(): use to generate shuffled training data.
"""
import numpy as np
import data_preparation
""" Applying PCA function on flow vector """ 
from sklearn.decomposition import PCA 

def flow_data_pca(flows,energyUsed):
    """
        This data_analysis function is used to analyze flows numpy array
        Arguments: 
            flows: input data with numpy type in the shape (number of data,features)
            energyUsed: (type double) the percent of energy to be preserved
        Returns:
            flows_reduced: reduce features in the flows
    """
    # assert(numKernels <= flows.shape[1])            # Error: number of kernels are more than flows' features
    # Do PCA 
    pca = PCA(n_components = flows.shape[1])
    pca.fit(flows)
    # print(pca.singular_values_)
    # print(pca.explained_variance_ratio_)
    energy = np.cumsum(pca.explained_variance_ratio_)
    # print(energy)
    numComponents = np.sum(energy < energyUsed) + 1
    # print(numComponents)
    
    pca = PCA(n_components = numComponents)
    
    return pca.fit_transform(flows) 

def OD_data_pca(ODs,energyUsed):
    """
        This data_analysis function is used to analyze OD matrix numpy array
        Arguments: 
            ODs: input data with numpy type in the shape (number of data,features)
            energyUsed: (type double) the percent of energy to be preserved
        Returns:
            ODs_reduced: reduce features in the ori ODs
    """
    # assert(numKernels <= ODs.shape[1])              # Error: number of kernels are more than flows' features
    # Do PCA
    pca = PCA(n_components = ODs.shape[1])
    pca.fit(ODs)
    # print(pca.singular_values_)
    # print(pca.explained_variance_ratio_)
    energy = np.cumsum(pca.explained_variance_ratio_)
    # print(energy)
    numComponents = np.sum(energy < energyUsed) + 1
    # print(numComponents)
    
    pca = PCA(n_components = numComponents)
    
    return pca.fit_transform(ODs) 

def training_data_generation(flows_reduced,ODs_reduced,numDays):
    """
        Function to generate training data:
            Reason: limit data-set
        Conditions: Generate data in a four data period. Result data size: (numOfData,4(days)*24(hours),PCA_kernels)
        Arguments:
            flows_reduced: flows_data after pca
            ODs_reduced: vectorized OD matrix data after pca
            numDays: time length of RNN model
        Returns:
            X_train_shuffled: a shuffled flows_reduced data with size (numOfData,4(days)*24(hours),PCA_kernels)
            Y_train_shuffled: a shuffled ODs_reduced data with size (numOfData,4(days)*24(hours),PCA_kernels)
    """
    # Define Constant:
    HOURSINADAY = 24                                            # Define constant: total hour in a day
    print(flows_reduced.shape)
    print(ODs_reduced.shape)

    assert(flows_reduced.shape[0] == ODs_reduced.shape[0])      # Number of Training Data does not match

    # Data generation
    flows_list = []
    ODs_list = []
    for i in range (0,flows_reduced.shape[0]-int(numDays*HOURSINADAY)):
        flows_list.append(flows_reduced[i:i+int(numDays*HOURSINADAY)])
        ODs_list.append(flows_reduced[i:i+int(numDays*HOURSINADAY)])
    # Change to numpy array
    X_train = np.asarray(flows_list)
    Y_train = np.asarray(ODs_list)

    assert(X_train.shape[0] == Y_train.shape[0])                # Generate Data has different dimension
    # Data shuffle
    pi = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[pi]
    Y_train_shuffled = Y_train[pi]

    return X_train_shuffled,Y_train_shuffled


# if __name__ == "__main__":
#     # Load File From Given Flow Directory
#     flows, flow_sequence = data_preparation.load_flow_directory_files('../data/TrainingFlow')

#     # Load File From Given OD Directory
#     vectorized_od = data_preparation.load_OD_directiory_files('../data/TrainingOD',flow_sequence,flows)

#     energyUsed = 0.98
#     flows_reduced = flow_data_pca(flows,energyUsed)

#     # numKernels_ODs = vectorized_od.shape[1]
#     ODs_reduced = OD_data_pca(vectorized_od,energyUsed)

# #     X_train,Y_train = training_data_generation(flows_reduced,ODs_reduced,4)

# #     print(X_train.shape)
# #     print(Y_train.shape)