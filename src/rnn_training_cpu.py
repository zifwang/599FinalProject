import numpy as np
import sys
import h5py
import data_preparation                 # Use load_flow_directory_files() & load_OD_directiory_files() functions. 
import util
from sklearn.model_selection import train_test_split
"""Keras"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.layers import CuDNNLSTM, CuDNNGRU

def create_rnn_model(rnnModel,rnn_type,inputSize,outputShape):
    """
        Function to create my rnn neural network
		Arguments: rnnModel: keras rnnModel
				   type: string input: choose model: GRU, LSTM
				   inputSize: training input size with shape (time_length,features)
                   outputShape: a training output shape (h,w,colorChannel) colorChannel should be 1 here
		Return: model after set up
    """
    # If doesn't given rnn_type and inputSize return false
    # if rnn_type and inputSize: 
    #     sys.exit()

    if(rnn_type == 'GRU'):
        rnnModel.add(GRU(units=32, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
                        recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, 
                        activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                        dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, 
                        stateful=False, unroll=False, reset_after=False, input_shape=inputSize))
    
    if(rnn_type == 'LSTM'):
        rnnModel.add(LSTM(units=32, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
                    recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, 
                    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                    dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, 
                    stateful=False, unroll=False, input_shape=inputSize))
    
    # Can try leakyRelu here
    rnnModel.add(Dense(128,activation='relu'))       
    rnnModel.add(Dense(64,activation='relu'))
    rnnModel.add(Dense(outputShape[1],activation='relu'))

    rnnModel.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])

    return rnnModel

def main():
    ##############################################################################################
    """
        Section of Loading training data
        Variable here:
            1. numKernels_flows: number of PCA kernels for flows data
            2. numKernels_ODs: number of PCA kernels for ODs data
            3. numDays: time length of training RNN
        Return:
            X_train: flows
            Y_train: vectorized_OD
    """
    # Variables Defines In here: change in the experiments
    numKernels_flows = 15
    numKernels_ODs = 15
    numDays = 4

    # Load File From Given Flow Directory
    flows, flow_sequence = data_preparation.load_flow_directory_files('../data/TrainingFlow')

    # Load File From Given OD Directory
    vectorized_od = data_preparation.load_OD_directiory_files('../data/TrainingOD',flow_sequence,flows)

    # Apply PCA method to reduce flows & vectorized_od data
    flows_reduced = util.flow_data_pca(flows,numKernels_flows)
    ODs_reduced = util.OD_data_pca(vectorized_od,numKernels_ODs)

    # Generate a shuffled X_train and its corresponding Y_train data
    X_train,Y_train = util.training_data_generation(flows_reduced,ODs_reduced,numDays)
    ##############################################################################################

    ##############################################################################################
    """
        Section of Building RNN
    """
    # Get training input size
    numOfX,time_length_x,features_x = X_train.shape
    numOfY,time_length_y,features_y = Y_train.shape
    assert(numOfX == numOfY)
    # # Reshape Y_train to the one whose size has a image like
    # Y_train = Y_train.reshape((numOfY,time_length_y,features_y,1))

    rnnModel = Sequential()
    rnnModel = create_rnn_model(rnnModel,'GRU',(time_length_x,features_x),(time_length_y,features_y))
    
    # Print model summary
    rnnModel.summary()
    
    # Training
    rnnModel.fit(x=X_train,
          y=Y_train, 
          batch_size=128, 
          epochs=50, 
          verbose=1,
          validation_split=0.1,
          shuffle=True
          )
    
    # Save Model
    rnnModel.save('../model/my_rnn_model.h5')
    print('Save Model to Disk')
    ##############################################################################################

if __name__ == '__main__':
    main()