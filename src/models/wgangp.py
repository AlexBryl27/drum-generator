from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv1DTranspose
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Layer, Embedding, Concatenate

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
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
        alpha = K.random_uniform((self.batch_size, 1, 1))
        return [
            (alpha * inputs[0][0]) + ((1 - alpha) * inputs[1][0]),
            (alpha * inputs[0][1]) + ((1 - alpha) * inputs[1][1])
        ]


class MusicGAN():
    
    def __init__(
        self,
        n_notes,
        n_durations,
        input_dim,
        kernel_size,
        critic_filters,
        critic_strides,
        generator_filters,
        generator_strides,
        grad_weight,
        z_dim,
        batch_size,
        learning_rate,
        use_batch_norm
    ):
        self.n_notes = n_notes
        self.n_durations = n_durations
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.critic_filters = critic_filters
        self.critic_strides = critic_strides
        self.generator_filters = generator_filters
        self.generator_strides = generator_strides
        self.grad_weight = grad_weight
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm

        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self._build_critic()
        self._build_generator()
        self._build_adversarial()


    def gradient_penalty_loss(self, y_true, y_pred, interpolated_sample):
        # gradients = grad(y_pred, interpolated_sample)[0]
        gradients = K.gradients(y_pred[0], interpolated_sample[0])[0]
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
        notes_in = Input(shape=(self.input_dim, self.n_notes))
        durations_in = Input(shape=(self.input_dim, self.n_durations))
        x = Concatenate()([notes_in, durations_in])
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
        self.dense_shape = x.shape[1:]
        x = Flatten()(x)
        critic_output = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)
        self.critic = Model([notes_in, durations_in], critic_output)

    
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

        self.set_trainable(self.generator, False)
        real_notes = Input(shape=(self.input_dim, self.n_notes))
        real_durations = Input(shape=(self.input_dim, self.n_durations))
        real_sequence = [real_notes, real_durations]
        valid = self.critic(real_sequence)
        
        z = Input(shape=(self.z_dim,))
        fake_sequence = self.generator(z)
        fake = self.critic(fake_sequence)

        interpolated_sequence = RandomWeightedAverage(self.batch_size)([real_sequence, fake_sequence])
        interpolated = self.critic(interpolated_sequence)

        gp_loss = partial(self.gradient_penalty_loss, interpolated_sample=interpolated_sequence)
        gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model([real_sequence, z], [valid, fake, interpolated])
        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, gp_loss],
            optimizer=Adam(lr=self.learning_rate),
            loss_weights=[1, 1, self.grad_weight]
        )


        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        model_input = Input(shape=(self.z_dim,))
        sequence = self.generator(model_input)
        model_output = self.critic(sequence)
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=self.wasserstein
        )

        self.set_trainable(self.critic, True)


    def train_critic(self, X, batch_size):

        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32)

        idx = np.random.randint(0, X[0].shape[0], batch_size)
        true_sequences = [X[0][idx], X[1][idx]]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        c_loss = self.critic_model.train_on_batch([true_sequences, noise], [valid, fake, dummy])
        return c_loss

    def train_generator(self, batch_size):    
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    
    def train(self, X, batch_size, epochs, n_critic=5, save_every_n_batches=10):
        self.c_losses, self.g_losses = [], []
        self.generated = {}
        for epoch in range(epochs):
            for _ in range(n_critic):
                c_loss = self.train_critic(X, batch_size)
            g_loss = self.train_generator(batch_size)
            print(f"{epoch} epoch. Critic losses: real - {c_loss[0]}, fake = {c_loss[1]}, penalty = {c_loss[2]} \
                Generator loss - {g_loss}")
            
            self.c_losses.append(c_loss)
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

