import numpy as np
import h5py
import data_preparation                 # Use load_flow_directory_files() & load_OD_directiory_files() functions. 
import util
from sklearn.model_selection import train_test_split
"""Keras"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.layers import CuDNNLSTM, CuDNNGRU