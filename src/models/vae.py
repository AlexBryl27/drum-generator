import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten, Dense, Lambda
from tensorflow.keras.layers import Conv1DTranspose, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np