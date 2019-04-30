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

if __name__ == "__main__":
    # Load File From Given Flow Directory
    flows, flow_sequence = data_preparation.load_flow_directory_files('../data/TrainingFlow')

    # Load File From Given OD Directory
    vectorized_od = data_preparation.load_OD_directiory_files('../data/TrainingOD',flow_sequence,flows)

    numKernels_flows = 10
    _ = flow_data_pca(flows,numKernels_flows)


    numKernels_ODs = 10
    _ = OD_data_pca(vectorized_od,numKernels_ODs)
