'''
## Network ##
# Defines the DQN network - architecture, training step and testing step. 
@author: Kolin Guo
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv3D
from tensorflow.keras import Model

class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Input dim: (batch, H, W, D, channels) = (32, 10, 10, 7, 4)
        self.conv1 = Conv3D(32, (7, 7, 5), 1, activation='relu')
        self.conv2 = Conv3D(64, (5, 5, 5), 1, activation='relu')
        self.conv3 = Conv3D(64, (3, 3, 3), 1, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(4)
