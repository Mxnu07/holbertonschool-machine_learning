#!/usr/bin/env python3
""" Wassertein GAN with gradient penalty """

import tensorflow as tf
from tensorflow import keras
import numpy as np


class WGAN_GP(keras.Model):
    """ Defines a WGAN with Gradient Penalty """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """ Initializes a WGAN with Gradient Penalty
            """
        super().__init__()  # Keras model is run first

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        # =================== New! ===================
        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')

        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1

        self.scal_shape = tf.convert_to_tensor(self.scal_shape)
        #  =================== New! ===================

        # Generator loss and optimizer
        rm = tf.reduce_mean
        self.generator.loss = lambda x: -rm(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # Discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: rm(y) - rm(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """ Generates a fake sample
        """
        if not size:
            size = self.batch_size

        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """ Retrieves a sample of real randomly sorted examples
        """
        if not size:
            size = self.batch_size

        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]

        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """ Fetches an interpolated sample
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u*real_sample + v * fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """ Gradient penalty with respect to an interpolated sample
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)

        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """ Performs one training step for the WGAN model """
        for _ in range(self.disc_iter):
            # Penalized loss for the discriminator

            with (tf.GradientTape() as disc_tape):
                # Real, fake and interpolated sample
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample()
                interpoled_sample = self.get_interpolated_sample(real_sample,
                                                                 fake_sample)

                # Old loss discr_loss of the discr on real and fake samples
                disc_real_output = self.discriminator(real_sample)
                disc_fake_output = self.discriminator(fake_sample)

                discr_loss = self.discriminator.loss(
                    x=disc_real_output,
                    y=disc_fake_output)

                # compute the gradient penalty gp
                gp = self.gradient_penalty(interpoled_sample)

                # new_discr_loss = discr_loss + self.lambda_gp * gp
                ndiscr_loss = discr_loss + self.lambda_gp * gp

            # Gradient descent with respect to new_discr_loss
            disc_gradients = (
                disc_tape.gradient(ndiscr_loss,
                                   self.discriminator.trainable_variables))

            self.discriminator.optimizer.apply_gradients(zip(
                disc_gradients,
                self.discriminator.trainable_variables))

        # Loss for the generator
        with tf.GradientTape() as gen_tape:
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)
            generator_output = self.discriminator(fake_sample, training=False)

            # compute the loss gen_loss of the generator on this sample
            generator_loss = self.generator.loss(generator_output)

        # Gradient descent applied to the discriminator (gp = gradient penalty)
        gen_gradients = gen_tape.gradient(generator_loss,
                                          self.generator.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(
            gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": generator_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """ Replace the weights for generator and discriminator
            by the ones stored in the .h5 files
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
