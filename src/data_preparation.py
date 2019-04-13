import numpy as np
import pandas as pd
""" Applying PCA function on flow vector """ 
from sklearn.decomposition import PCA 




def load_data_file(fileName):
    """
        This function is used to load data from file using pandas library
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
     



    # Drop two unnecessary rows
    df = df.drop(['Unnamed: 0','Unnamed: 36'], axis = 1)            # remove the first and the last column from the data frame
    # create a flow list
    flows = []
    for i in df.index.values:
        index_array = np.asarray(df.loc[[i]])
        flows.append(index_array.T)
    # change to numpy array
    flows = np.asarray(flows)
    samples,features,_ = flows.shape                        # get shape
    # reshape flows array
    flows = np.reshape(flows,(samples,features))

    return flows

def data_cleaning(df):
    """
        This function is used to clean data frame
        Arguments: df: pd dataframe
        Returns: dataFrame: a clean data frame with no repeat elements
    """

    
    


def data_analysis(flows):
    # See the relationship in 24 hours 
    flows_24 = []
    # for i in range (24):
    #     index = np.asarray(df.loc[[i]])
    #     flows_24.append(index.T)
    # flows_24_np = np.asarray(flows_24)                                 # shape = (24,35,1)
    # # flows_24_np means 24 flow vectors in each hour of a day and 35 features in each flow vector
    # flows_24_np = np.reshape(flows_24_np,(24,35))                      # reshape to (24,35)
    # # PCA 
    # pca = PCA(n_components = 24)
    # pca.fit(flows_24_np)
    # print(pca.singular_values_)



flows = load_data_file('../data/TrainingFlow/flow_OD1_0-239.csv')
# flows = load_data_file('../data/TrainingFlow/flow_OD1_240-719.csv')