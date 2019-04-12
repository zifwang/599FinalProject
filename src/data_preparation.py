import numpy as np
import pandas as pd
""" Applying PCA function on flow vector """ 
from sklearn.decomposition import PCA 




def load_data_file(fileName):
    """
        This function is used to load data from file using pandas library
        Arguments: fileName: The .csv file name
        Returns:
    """
    df = pd.read_csv(fileName)
    df = df.drop(['Unnamed: 0','Unnamed: 36'], axis = 1)            # remove the first and the last column from the data frame
    indexed_df = df.loc[[0]]
    index = np.asarray(indexed_df)
    print(index.shape)
    # pca = PCA(n_components = 34)
    # pca.fit(index)
    # print(pca.singular_values_)
    
    # See the relationship in 24 hours 
    flows_24 = []
    for i in range (24):
        index = np.asarray(df.loc[[i]])
        flows_24.append(index)
    flows_24_np = np.asarray(flows_24)
    print(flows_24_np.shape)

load_data_file('../data/TrainingFlow/flow_OD1_0-239.csv')