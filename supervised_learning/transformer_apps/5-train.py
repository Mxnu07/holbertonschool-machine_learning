#!/usr/bin/env python3
"""
Training a full Transformer network
"""

import tensorflow as tf
import keras as K
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(K.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler based on the formula.
    """
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Loss function that ignores padding tokens.
    """
    loss_object = K.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(tf.not_equal(real, 0), dtype=tf.float32)  # Assume 0 is the padding token
    loss_ = loss_object(real, pred)

    loss_ *= mask  # Only consider non-padding tokens for the loss

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Trains a transformer model for Portuguese to English translation.
    """
    # Load dataset
    data = Dataset(batch_size, max_len)
    data_train = data.data_train
    data_valid = data.data_valid

    # Initialize transformer model
    model = Transformer(N, dm, h, hidden,
                        input_vocab=data.tokenizer_pt.vocab_size + 2,  # +2 for special tokens
                        target_vocab=data.tokenizer_en.vocab_size + 2,
                        max_seq_input=max_len,
                        max_seq_target=max_len,
                        drop_rate=0.1)

    # Custom learning rate schedule and Adam optimizer
    learning_rate = CustomSchedule(dm)
    optimizer = K.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Metrics to track progress
    train_loss = K.metrics.Mean(name='train_loss')
    train_accuracy = K.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # Define the training step as a tf.function for performance
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = model(inp, tar_inp, True, enc_mask, combined_mask, dec_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} started...')

        # Reset metrics at the start of every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Iterate over batches in training data
        for (batch, (inp, tar)) in enumerate(data_train):
            train_step(inp, tar)

            # Print progress every 50 batches
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch}: Loss {train_loss.result()}, '
                      f'Accuracy {train_accuracy.result()}')

        # Print epoch-level summary
        print(f'Epoch {epoch + 1} completed. Loss: {train_loss.result()}, '
              f'Accuracy: {train_accuracy.result()}')

    return model
