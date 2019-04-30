import numpy as np
import data_preparation
""" Applying PCA function on flow vector """ 
from sklearn.decomposition import PCA 

def flow_data_pca(flows,numKernels):
    """
        This data_analysis function is used to analyze flows numpy array
        Arguments: 
            flows: input data with numpy type in the shape (number of data,features)
            numkernels: number of kernels used in PCA (constrained here make sure numKernels less than )
        Returns:
            flows_reduced: reduce features in the flows
    """
    assert(numKernels <= flows.shape[1])            # Error: number of kernels are more than flows' features
    # Do PCA 
    pca = PCA(n_components = numKernels)
    pca.fit(flows)
    print(pca.singular_values_)
    print(pca.explained_variance_ratio_)
    
    return pca.transform(flows)

def OD_data_pca(ODs,numKernels):
    """
        This data_analysis function is used to analyze OD matrix numpy array
        Arguments: 
            ODs: input data with numpy type in the shape (number of data,features)
            numkernels: number of kernels used in PCA (constrained here make sure numKernels less than )
        Returns:
            ODs_reduced: reduce features in the ori ODs
    """
    assert(numKernels <= ODs.shape[1])              # Error: number of kernels are more than flows' features
    # Do PCA
    pca = PCA(n_components = numKernels)
    pca.fit(ODs)
    print(pca.singular_values_)
    print(pca.explained_variance_ratio_)
    
    return pca.transform(ODs)

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
    
    assert(flows_reduced.shape[0] == ODs_reduced.shape[0])      # Number of Training Data does not match

    # Data generation
    flows_list = []
    ODs_list = []
    for i in range (0,flows_reduced.shape[0]-numDays*HOURSINADAY):
        flows_list.append(flows_reduced[i:i+numDays*HOURSINADAY])
        ODs_list.append(flows_reduced[i:i+numDays*HOURSINADAY])
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

#     numKernels_flows = 15
#     flows_reduced = flow_data_pca(flows,numKernels_flows)

#     numKernels_ODs = 15
#     ODs_reduced = OD_data_pca(vectorized_od,numKernels_ODs)

#     X_train,Y_train = training_data_generation(flows_reduced,ODs_reduced,4)

#     print(X_train.shape)
#     print(Y_train.shape)