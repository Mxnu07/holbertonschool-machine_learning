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
        # Load the VGG19 model for Neural Style Transfer
        self.load_model()
        # Generate the style features and content feature
        self.generate_features()

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
        # Initialize VGG19 -- the base model, excluding the
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

        # Gather outputs frm layers specified for capturing style
        # These layers are predefined and crucial for extracting style features
        style_outputs = \
            [vgg.get_layer(name).output for name in self.style_layers]
        # Similarly, capture the output frm the designated content layer
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
        # that captures the correlations between different channels=
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

    def generate_features(self):
        """"Extracts features used to calculate
        Neural Style Transfer cost
        Sets the public instance attributes:
            gram_style_features - a list of gram matrices
                calculated frm the style layer outputs
            content_feature - the content later output
                of the content image
        """
        # Preprocess the style image and content image
        # using the VGG19 model
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Extract the style and content features
        # frm the respective image
        style_outputs = self.model(style_image)
        content_output = self.model(content_image)

        # Extract the style layer outputs
        style_features = style_outputs[:-1]
        # Extract the content layer output
        content_feature = content_output[-1]

        # Calculate the gram matrices for each style layer
        self.gram_style_features = [NST.gram_matrix(style_feature)
                                    for style_feature in style_features]
        # Set the content feature
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer
        Arguments:
            style_output {tf.Tensor} -- the layer style output
            gram_target {np.ndarray} -- the gram matrix of the
            target style
        Returns:
            tf.Tensor -- the layer style cost
        """
        # Extract the number of channels frm the ]
        # last dimension of style_output
        c = style_output.shape[-1]

        # Validate that style_output is a 4D tensor
        # (batch, height, width, channels)
        err_1 = "style_output must be a tensor of rank 4"
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err_1)
        if len(style_output.shape) != 4:
            raise TypeError(err_1)

        # Validate that gram_target is a 3D tensor
        # with shape [1, channels, channels]
        # This shape ensures that gram_target is a
        # square matrix representing
        # the correlations between channels of a
        # single style feature map
        err_2 = \
            ("gram_target must be a tensor of shape [1, {}, {}]".format(c, c))
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(err_2)
        if gram_target.shape != (1, c, c):
            raise TypeError(err_2)

        # Compute the gram matrix of the style_output
        # layer using the gram_matrix method
        # The gram matrix represents the inner products
        # between the vectorized feature maps,
        # capturing the style information of the image
        gram_style = self.gram_matrix(style_output)

        # Calculate the mean squared error between the c
        # omputed gram matrix (gram_style)
        # and the target gram matrix (gram_target). This
        # error represents the style cost,
        # quantifying how much the style of the generated
        # image deviates frm the target style
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return style_cost

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image
        Arguments:
            style_outputs {list} -- a list of tf.Tensor
            containing the style outputs for the generated image
        Returns:
            tf.Tensor -- the style cost
        """
        # Validate that style_outputs is a list and its length
        # matches the number of style layers
        # This ensures that there is a one-to-one correspondence
        # between the layers specified
        # for style representation and the actual outputs frm
        # the style image
        st_len = len(self.style_layers)
        err_list_check = \
            "style_outputs must be a list with a length of {}".format(st_len)
        if not isinstance(style_outputs, list):
            raise TypeError(err_list_check)
        if len(self.style_layers) != len(style_outputs):
            raise TypeError(err_list_check)

        # Reminders:
        # style_layers is a list of name strings indicating
        # which layers of the network
        # are used to compute the style representation
        # style_outputs is a list of tensors containing the
        # activations of the style layers
        # for the generated image

        # Initialize a list to hold the style costs for each layer
        style_costs = []
        # Each layer's style cost is weighted evenly, with all weights
        # summing to 1.
        # This ensures that no single layer disproportionately ]
        # influences the total style cost.
        weight = 1 / len(self.style_layers)

        # Calculate the weighted style cost for each layer
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            # Compute the style cost for the current layer
            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            # Apply the weight to the layer's style cost
            weighted_layer_style_cost = weight * layer_style_cost
            # Add the weighted style cost to the list of style costs
            style_costs.append(weighted_layer_style_cost)

        # Sum all the weighted style costs to get the total style cost
        # tf.add_n is used to sum a list of tensors, producing a single tensor
        style_cost = tf.add_n(style_costs)

        return style_cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image
        Arguments:
            content_output {tf.Tensor} -- the content output
            for the generated image
        Returns:
            tf.Tensor -- the content cost
        """
        # Reminder:
        # The content layer output of the content_image is
        # stored in self.content_feature.
        # This tensor captures the high-level content
        # details of the content image.

        # Ensure content_output matches the expected 4D shape
        # of self.content_feature.
        # If content_output is 3D (height, width, channels),
        # add a batch dimension at the start.
        if content_output.ndim == 3:
            # Add batch dimension
            content_output = content_output[tf.newaxis, ...]

        # Validate that content_output has the same shape --
        # self.content_feature.
        # This ensures that the comparison between the generated
        # image and the content image
        # is valid and meaningful.
        cn_fe = self.content_feature.shape
        err_shape_check = \
            "content_output must be a tensor of shape {}".format(cn_fe)
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err_shape_check)
        if content_output.shape != self.content_feature.shape:
            raise TypeError(err_shape_check)
        if content_output.shape[-1] != 512:
            raise TypeError(err_shape_check)

        # Compute the content cost -- the mean squared error
        # between content_output
        # and self.content_feature. This cost function encourages
        # the generated image
        # to have similar content features to the content image,
        # effectively transferring
        # the content of the content image to the generated image
        # while retaining the style
        # of the style image.1
        content_cost = tf.reduce_mean(
            tf.square(content_output - self.content_feature))

        return content_cost

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image
        Arguments:
            generated_image {tf.Tensor} -- the generated image
        Returns:
            tuple -- (J, J_content, J_style)
                J - the total cost
                J_content - the content cost
                J_style - the style cost
        """
        # Validate the input image's type and shape
        validation_error = \
            "generated_image must be a tensor of shape {}".format(
                self.content_image.shape)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(validation_error)
        if generated_image.shape != self.content_image.shape:
            raise TypeError(validation_error)

        # Preprocess the input image for the neural network
        # Adjust pixel values to the expected range for VGG19
        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        # Obtain both style and content features frm the model
        model_outputs = self.model(generated_image)
        # Separate style features frm the model outputs
        style_features = model_outputs[:-1]
        # Separate content feature frm the model outputs
        content_feature = model_outputs[-1]

        # Compute the style loss using the extracted style features
        style_loss = self.style_cost(style_features)
        # Compute the content loss using the extracted content feature
        content_loss = self.content_cost(content_feature)

        # Calculate the total loss as a weighted sum
        # of content and style losses
        total_loss = (self.alpha * content_loss + self.beta * style_loss)

        return (total_loss, content_loss, style_loss)
