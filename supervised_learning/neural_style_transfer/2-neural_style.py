#!/usr/bin/env python3
"""This module contain the clas NST
"""
import numpy as np
import tensorflow as tf


class NST:
    """This is the class NST"""

    # Public class attributes
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layer = "block5_conv2"

    # Class constructor
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializer
        Arguments:
            style_image {np.ndarray} -- the image style
            content_image {np.ndarray} -- the image content
            alpha {float} -- the weight for style cost
            beta {float} -- the weight for content cost
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be "
        error2 += "a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3:
            raise TypeError(error1)
        if style_image.shape[-1] != 3:
            raise TypeError(error1)
        if not isinstance(content_image,
                          np.ndarray) or content_image.ndim != 3:
            raise TypeError(error2)
        if content_image.shape[-1] != 3:
            raise TypeError(error2)
        if not (isinstance(alpha, (float, int)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (float, int)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales the image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        Arguments:
            image {np.ndarray} -- the image to be scaled
        Returns:
            np.ndarray -- the scaled image
        """
        error = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError(error)
        if image.shape[-1] != 3:
            raise TypeError(error)
        max_dim = 512
        h, w, _ = image.shape
        scale = max_dim / max(h, w)
        h_new, w_new = int(h * scale), int(w * scale)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, [h_new, w_new], method="bicubic")
        image /= 255.0
        image = tf.clip_by_value(image, 0, 1)
        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Loads the model for Neural Style Transfer"""
        # Initialize VGG19 as the base model, excluding the
        # top layer (classifier)
        # The model uses the default input size of 224x224 pixels
        base_vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )
        # Modify the model by substituting MaxPooling with
        # AveragePooling
        # Achieved by utilizing the custom_objects parameter
        # during model loading
        # This modification can enhance the quality of features
        # extracted for NST
        custom_object = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        base_vgg.save("base_vgg")
        # Reload the VGG model with the pooling layers swapped
        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=custom_object)
        # Directly setting vgg.trainable to False is ineffective
        # Instead, set each layer's trainable attribute
        # to False to lock weights
        # Prevents the alteration of pre-trained weights
        # during the NST process
        for layer in vgg.layers:
            layer.trainable = False

        # Gather outputs from layers specified for capturing style
        # These layers are predefined and crucial for extracting style features
        style_outputs = \
            [vgg.get_layer(name).output for name in self.style_layers]
        # Similarly, capture the output from the designated content layer
        # This layer is pivotal for content feature extraction
        content_output = vgg.get_layer(self.content_layer).output
        # Merge style and content layer outputs for comprehensive
        # feature representation
        # This concatenated output facilitates simultaneous style
        # and content optimization
        outputs = style_outputs + [content_output]
        # Construct a new model tailored for NST by specifying
        # desired inputs and outputs
        # This custom model is central to the NST algorithm,
        # enabling feature extraction
        # The model is stored for subsequent use in the NST process
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer
        Arguments:
            input_layer {tf.Tensor} -- the layer for which to calculate
            the gram matrix
        Returns:
            tf.Tensor -- the gram matrix
        """
        error = "input_layer must be a tensor of rank 4"
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(error)
        if len(input_layer.shape) != 4:
            raise TypeError(error)

        # Compute the outer product of the input tensor (feature map)
        # input_layer with itself using tf.linalg.einsum.
        # This function performs
        # tensor operations using Einstein summation convention.
        # The 'bijc,bijd->bcd'
        # notation specifies the operation: 'bijc' and 'bijd'
        # represent the dimensions
        # of the input tensors, indicating batch size (b),
        # height (i), width (j),
        # and channels (c or d). The '->bcd' specifies the
        # output dimensions, where
        # the operation sums over the 'i' and 'j' dimensions,
        # resulting in a tensor
        # that captures the correlations between different channels
        # across all spatial
        # locations, effectively computing the Gram matrix for style
        # representation.
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Average over all the elements ("pixels" or "grid cell") of
        # the feature map to normalize the result. This division
        # by the number of
        # locations ensures that the scale of the Gram matrix
        # does not depend on the
        # size of the input image.
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations
