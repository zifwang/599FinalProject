import numpy as np
""" Applying PCA function on flow vector """ 
from sklearn.decomposition import PCA 


def flow_data_pca(flows,numKernels):
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

def OD_data_pca(ODs,numKernels):
    """
        This data_analysis function is used to analyze OD matrix numpy array
        Arguments: 
            ODs: input data with numpy type in the shape (number of data,features)
            numkernels: number of kernels used in PCA (constrained here make sure numKernels less than )
        Returns:
    """
    