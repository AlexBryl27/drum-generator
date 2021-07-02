from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv1DTranspose
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Embedding, Concatenate

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

import numpy as np


class MusicGAN():
    
    def __init__(
        self,
        n_notes,
        n_durations,
        input_dim,
        kernel_size,
        discriminator_filters,
        discriminator_strides,
        generator_filters,
        generator_strides,
        z_dim,
        learning_rate,
        use_batch_norm
    ):
        self.n_notes = n_notes
        self.n_durations = n_durations
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.discriminator_filters = discriminator_filters
        self.discriminator_strides = discriminator_strides
        self.generator_filters = generator_filters
        self.generator_strides = generator_strides
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm

        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self._build_discriminator()
        self._build_generator()
        self._build_adversarial()


    def _build_discriminator(self):
        notes_in = Input(shape=(self.input_dim, self.n_notes))
        durations_in = Input(shape=(self.input_dim, self.n_durations))
        x = Concatenate()([notes_in, durations_in])
        self.inp_shape = x.shape[1]
        for i in range(len(self.discriminator_filters)):
            x = Conv1D(
                filters=self.discriminator_filters[i], 
                kernel_size=self.kernel_size, 
                strides=self.discriminator_strides[i], 
                padding='same',
                kernel_initializer=self.weight_init
                )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        self.dense_shape = x.shape[1:]
        x = Flatten()(x)
        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)
        self.discriminator = Model([notes_in, durations_in], discriminator_output)

    
    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim,))
        x = Dense(np.prod(self.dense_shape), kernel_initializer=self.weight_init)(generator_input)
        x = Reshape(self.dense_shape)(x)
        for i in range(len(self.generator_filters)):
            x = Conv1DTranspose(
                filters=self.generator_filters[i],
                kernel_size=self.kernel_size,
                strides=self.generator_strides[i],
                padding='same',
                kernel_initializer=self.weight_init
            )(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        x_notes, x_durations = tf.unstack(Reshape((2, self.inp_shape, self.generator_filters[-1] // 2))(x), axis=1)
        notes_out = Dense(self.n_notes, activation='tanh')(x_notes)
        durations_out = Dense(self.n_durations, activation='tanh')(x_durations)
        self.generator = Model(generator_input, [notes_out, durations_out])

    
    def set_trainable(self, model, val):
        model.trainable = val
        for layer in model.layers:
            layer.trainable = val

    def _build_adversarial(self):

        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=self.learning_rate)
        )

        self.set_trainable(self.discriminator, False)
        model_input = Input(shape=(self.z_dim,))
        sequence = self.generator(model_input)
        model_output = self.discriminator(sequence)
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss='binary_crossentropy',
            experimental_run_tf_function=False
        )
        self.set_trainable(self.discriminator, True)


    def train_discriminator(self, X, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = np.zeros((batch_size, 1), dtype=np.float32)

        idx = np.random.randint(0, X[0].shape[0], batch_size)
        true_sequences = [X[0][idx], X[1][idx]]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        generated_sequence = self.generator.predict(noise)
        d_loss_real = self.discriminator.train_on_batch(true_sequences, valid)
        d_loss_fake = self.discriminator.train_on_batch(generated_sequence, fake)
        return d_loss_real, d_loss_fake

    def train_generator(self, batch_size):    
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    
    def train(self, X, batch_size, epochs, save_every_n_batches=10):
        self.c_losses, self.g_losses = [], []
        self.generated = {}
        for epoch in range(epochs):
            d_loss_real, d_loss_fake = self.train_discriminator(X, batch_size)
            g_loss = self.train_generator(batch_size)
            print(f"{epoch} epoch. discriminator losses: real - {d_loss_real}, fake = {d_loss_fake}, \
                Generator loss - {g_loss}")
            
            self.c_losses.append([d_loss_real, d_loss_fake])
            self.g_losses.append(g_loss)

            if epoch % save_every_n_batches == 0:
                self.generated[epoch] = self.generate_sequence()


    def sample(self, preds, temperature):
        preds = np.where(preds < 0, 0, preds)
        preds += 1e-12
        if temperature == 0:
            return np.argmax(preds)
        else:
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            preds = preds[~np.isnan(preds)]
            return np.random.choice(len(preds), p=preds)


    def generate_sequence(self, temperatures=[0.0, 0.5, 0.9]):
        generated = {}
        noise = np.random.normal(0, 1, (1, self.z_dim))

        for temperature in temperatures:
            generated_notes = []
            generated_durations = []
            predicted_notes, predicted_durations = self.generator.predict(noise)
            for i in range(predicted_notes.shape[1]):
                generated_notes.append(self.sample(predicted_notes[0][i], temperature))
                generated_durations.append(self.sample(predicted_durations[0][i], temperature))
            generated[temperature] = [generated_notes, generated_durations]
        return generated
