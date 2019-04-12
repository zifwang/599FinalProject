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
    pca = PCA(n_components = 34)
    pca.fit(index)
    print(pca.singular_values_)

    # for i in df.index.values:
    #     df.loc[[i]

load_data_file('../data/TrainingFlow/flow_OD1_0-239.csv')