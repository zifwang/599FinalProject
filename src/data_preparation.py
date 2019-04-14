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
    df = data_cleaning(df)

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

def data_cleaning(df):
    """
        This function is used to clean data frame
        Arguments: df: pd dataframe
        Returns: dataFrame: a clean data frame with no repeat elements
    """
    # sort by hour
    df = df.sort_values(by=['Unnamed: 0'])
    # Check repeat and remove repeating hour
    hours = []                                                # create a list to contain hours
    for i in df.index.values:
        hour = int(df.loc[[i]]['Unnamed: 0'])                 # get hour 
        if hour not in hours:
            hours.append(hour)
        else:
            df = df.drop([i])
    
    return df
        

def data_analysis(flows,days):
    """
        This data_analysis function is used to analyze flows numpy array
        Arguments: 
            flows: input data with numpy type in the shape (number of data,features)
            days: number of days to see the relationship
        Returns:
    """
    # Define Constant here
    # days in a month
    DAYSINMONTH = 31 
    # hours in a day
    HOURSINDAY = 24

    # Define variable: total hours of given days
    total_hours = days*HOURSINDAY

    # Data Seperation
    flows_given_days = []           # a list contains data
    for i in range (0,DAYSINMONTH-days):
        flows_given_days.append(flows[i*HOURSINDAY:(i+days)*HOURSINDAY,:])
    
    # PCA 
    flows_dimension_reduction = []
    pca_score = []
    pca_precision = []
    minimum = min(total_hours,flows.shape[1])                        # Compare hours with features 
    pca = PCA(n_components = minimum)                                # Set up PCA by n_components
    start_day = 1
    end_day = 4
    for flow in flows_given_days:
        pca.fit(flow)
        pca_score.append(pca.score(flow))
        pca_precision.append(pca.get_precision())
        flows_dimension_reduction.append(pca.transform(flow))
        # print singular_values_
        print('Day ' + str(start_day) + '-' + str(end_day) + ': ')
        print(pca.singular_values_)
        start_day = start_day+1
        end_day = end_day+1

    


if __name__ == "__main__":
    # load file
    flows_1 = load_data_file('../data/TrainingFlow/flow_OD1_0-239.csv')
    flows_2 = load_data_file('../data/TrainingFlow/flow_OD1_240-719.csv')
    flows_ls = np.append(flows_1,flows_2,axis=0)                          # append flows together
    # try different input days below to see pca result
    print('print singular values:')
    data_analysis(flows_ls,4)
