from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv1DTranspose, 
from tensorflow.keras.layers import Reshape, Lambda, Activation, BatchNormalization, 
from tensorflow.keras.layers import LeakyReLU, Layer, Embedding, Concatenate

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
from tensorflow.python.keras.backend import shape
tf.compat.v1.disable_eager_execution()

from functools import partial
import numpy as np


def grad(x, y):
    v = Lambda(lambda z: K.gradients(
        z[0], z[1]), output_shape=[1])([x, y])
    return v

class RandomWeightedAverage(Layer):
    
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def call(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class MusicGAN():
    
    def __init__(
        self,
        n_notes,
        n_durations,
        emb_size,
        input_dim,
        kernel_size,
        critic_filters,
        critic_strides,
        critic_lr,
        generator_input_dim,
        generator_filters,
        generator_strides,
        generator_lr,
        grad_weight,
        z_dim,
        batch_size,
        use_batch_norm
    ):
        self.n_notes = n_notes,
        self.n_durations = n_durations,
        self.emb_size = emb_size,
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.critic_filters = critic_filters
        self.critic_strides = critic_strides
        self.critic_lr = critic_lr
        self.generator_input_dim = generator_input_dim
        self.generator_filters = generator_filters
        self.generator_strides = generator_strides
        self.generator_lr = generator_lr
        self.grad_weight = grad_weight
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.use_batch_norm = use_batch_norm

        self.weight_init = RandomNormal(mean=0., stddev=0.02)


    def gradient_penalty_loss(self, y_true, y_pred, interpolated):
        gradients = grad(y_pred, interpolated)[0]
        gradients_l2_norm = K.sqrt(
            K.sum(
                K.square(gradients),
                axis=np.arange(1,len(gradients.shape))
            )
        )
        gradient_penalty = K.square(1 - gradients_l2_norm)
        return K.mean(gradient_penalty)

    def wasserstein(self, y_true, y_pred):
        return - K.mean(y_true * y_pred)

    def _build_critic(self):
        notes_in = Input(shape=(self.input_dim))
        x_notes = Embedding(self.n_notes, self.emb_size)(notes_in)
        durations_in = Input(shape=(self.input_dim))
        x_durations = Embedding(self.n_durations, self.emb_size)(durations_in)
        x = Concatenate()([x_notes, x_durations])
        self.inp_shape = x.shape[1]
        for i in range(len(self.critic_filters)):
            x = Conv1D(
                filters=self.critic_filters[i], 
                kernel_size=self.kernel_size, 
                strides=self.critic_strides[i], 
                padding='same',
                kernel_initializer=self.weight_init
                )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        x = Flatten()(x)
        critic_output = Dense(1, activation=None, kernel_initializer=self.weight_init)(x)
        self.critic = Model([notes_in, durations_in], critic_output)

    
    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim,))
        x = generator_input
        x = Dense(np.prod(self.generator_input_dim), kernel_initializer=self.weight_init)(x)
        x = Reshape(self.generator_input_dim)(x)
        for i in range(len(self.generator_filters)):
            x = Conv1DTranspose(
                filters=self.generator_filters[i],
                kernel_size=self.kernel_size,
                strides=self.generator_strides[i],
                padding='same',
                kernel_initializer=self.weight_init
            )
            if i < len(self.generator_filters) - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
            else:
                continue
        x_notes, x_durations = tf.unstack(Reshape((2, self.inp_shape, self.emb_size))(x), axis=1)
        notes_out = Dense(self.n_notes, activation='tanh')(x_notes)
        durations_out = Dense(self.n_durations, activation='tanh')(x_durations)
        self.generator = Model(generator_input, [notes_out, durations_out])

    