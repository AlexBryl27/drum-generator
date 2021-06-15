import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense, Lambda
from tensorflow.keras.layers import Conv1DTranspose, Embedding, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

import numpy as np


class ModelBuilder(Model):
    def __init__(self, encoder, decoder, loss_factor):
        super(ModelBuilder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_factor = loss_factor
        self.cce = CategoricalCrossentropy()
    
    def compute_loss(self, x, y):
        mu, log_var, z = self.encoder(x)
        generated = self.decoder(z)
        predictions = generated[0]
        entropy = self.cce(y, predictions)
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis = 1) * self.loss_factor
        return entropy, kl_loss

    @tf.function
    def train_step(self, inputs):
        x = inputs[0]
        y = inputs[1][0]
        with tf.GradientTape() as tape:
            entropy, kl_loss = self.compute_loss(x, y)
            loss = entropy + kl_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {
            "cce": entropy,
            "kl": kl_loss,
            "loss": loss
        }

    def call(self, inputs):
        z = self.encoder(inputs)
        return self.decoder(z)


class MusicVAE:
    
    def __init__(
        self, 
        kernel_size,
        encoder_filters,
        encoder_strides,
        decoder_filters,
        decoder_strides,
        n_notes,
        n_durations,
        emb_size,
        input_dim,
        latent_dim,
        loss_factor,
        use_batch_norm = False
    ):
        self.kernel_size = kernel_size
        self.encoder_filters = encoder_filters
        self.encoder_strides = encoder_strides
        self.decoder_filters = decoder_filters
        self.decoder_strides = decoder_strides
        self.n_notes = n_notes
        self.n_durations = n_durations
        self.emb_size = emb_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.loss_factor = loss_factor
        self.use_batch_norm = use_batch_norm
        
        self._build_model()
        
        
    def _build_model(self):
        
        # ENCODER
        
        notes_in = Input(shape=(self.input_dim))
        x_notes = Embedding(self.n_notes, self.emb_size)(notes_in)
        durations_in = Input(shape=(self.input_dim))
        x_durations = Embedding(self.n_durations, self.emb_size)(durations_in)
        x = Concatenate()([x_notes, x_durations])
        inp_shape = x.shape[1]
        for i in range(len(self.encoder_filters)):
            x = Conv1D(self.encoder_filters[i], self.kernel_size, self.encoder_strides[i], padding='same')(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        dense_shape = x.shape[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_dim)(x)
        self.log_var = Dense(self.latent_dim)(x)
        
        def sample(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu))
            return mu + K.exp(log_var / 2) * epsilon
        
        self.z = Lambda(sample)([self.mu, self.log_var])
        self.encoder = Model([notes_in, durations_in], [self.mu, self.log_var, self.z])
        
        # DECODER
        
        decoder_input = Input(self.latent_dim)
        x = Dense(np.prod(dense_shape))(decoder_input)
        x = Reshape(dense_shape)(x)
        for i in range(len(self.decoder_filters)):
            x = Conv1DTranspose(self.decoder_filters[i], self.kernel_size, self.decoder_strides[i], padding='same')(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        x_notes, x_durations = tf.unstack(Reshape((2, inp_shape, self.emb_size))(x), axis=1)
        notes_out = Dense(self.n_notes, activation='softmax')(x_notes)
        durations_out = Dense(self.n_durations, activation='softmax')(x_durations)
        self.decoder = Model(decoder_input, [notes_out, durations_out])
        self.model = ModelBuilder(self.encoder, self.decoder, self.loss_factor)
        
    def compile(self, lr):
        self.model.compile(optimizer=Adam(lr))
