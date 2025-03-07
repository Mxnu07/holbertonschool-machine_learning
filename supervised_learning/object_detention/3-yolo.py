#!/usr/bin/env python3
"""This module contains the Yolo class
that uses the Yolo v3 algorithm to perform object detection
includes processing output method
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.activations import sigmoid  # type: ignore


class Yolo:
    """This class uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Yolo class constructor
        Args:
            model_path: is the path to where a Darknet Keras model is stored
            classes_path: is the path to where the list of class names used
                for the Darknet model, listed in order of index, can be found
            class_t: is a float representing the box score threshold for the
                initial filtering step
            nms_t: is a float representing the IOU threshold for non-max
                suppression
            anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes:
                outputs: is the number of outputs (predictions) made by the
                    Darknet model
                anchor_boxes: is the number of anchor boxes used for each
                    prediction
                2: [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        # Open file and read content
        with open(classes_path, "r") as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs: list of numpy.ndarrays containing the predictions from the
                Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width: the height and width of the
                    grid used
                        for the output
                    anchor_boxes: the number of anchor boxes used
                    4: (t_x, t_y, t_w, t_h)
                    1: box_confidence
                    classes: class probabilities for all classes
            image_size: numpy.ndarray containing the image’s original size
                [image_height, image_width]
        Returns: tuple of (boxes, box_confidences, box_class_probs):
            boxes: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, 4) containing the processed boundary boxes
                for each
                output, respectively:
                4: (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                    original image
            box_confidences: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, 1) containing the box confidences for
                each output,
                respectively
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, classes) containing the box’s class
                probabilities for
                each output, respectively
        """
        # List to hold the processed boundary boxes for each output
        boxes = []
        # List to hold the confidence for each box in each output
        box_confidences = []
        # List to hold the class probabilities for each box in each output
        box_class_probs = []

        # Unpack the outputs
        for i, output in enumerate(outputs):
            # Ig nore the rest with _
            grid_height, grid_width, anchor_boxes, _ = output.shape
            # Extract the box parameters
            box = output[..., :4]
            # Extract the individual components
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            # Create 3D grid for the anchor boxes
            # Create a grid for the x coordinates
            c_x = np.arange(grid_width).reshape(1, grid_width)
            #  Repeat the x grid anchor_boxes times
            c_x = np.repeat(c_x, grid_height, axis=0)
            # Reshape to add the anchor boxes
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            # Create a grid for the y coordinates
            c_y = np.arange(grid_width).reshape(1, grid_width)
            # Repeat the y grid anchor_boxes times
            c_y = np.repeat(c_y, grid_height, axis=0).T
            # Reshape to add the anchor boxes
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            # Create a grid for the anchor boxes
            b_x = (sigmoid(t_x) + c_x) / grid_width
            b_y = (sigmoid(t_y) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            # top left corner
            x1 = (b_x - b_w / 2)
            y1 = (b_y - b_h / 2)

            # bottom right corner
            x2 = (b_x + b_w / 2)
            y2 = (b_y + b_h / 2)

            # box coordinate relative to the image size
            x1 = x1 * image_size[1]
            y1 = y1 * image_size[0]
            x2 = x2 * image_size[1]
            y2 = y2 * image_size[0]

            # Update boxes
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            # Append the box to the boxes list
            boxes.append(box)

            # Extract the box confidence and aply sigmoid
            box_confidence = output[..., 4:5]
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_confidences.append(box_confidence)

            # Extract the box class probabilities and aply sigmoid
            box_class_prob = output[..., 5:]
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter the processed boundary boxes
        Args:
            boxes: list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 4) containing the processed
            boundary boxes
            box_confidences: list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box’s class
            probabilities for each output
        Returns: tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes:
                4: (x1, y1, x2, y2)
            box_classes: numpy.ndarray of shape (?,) containing the class
            number that each box in filtered_boxes predicts
            box_scores: numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes
        """

        box_scores = []
        box_classes = []
        filtered_boxes = []

        # Loop over the output feature maps
        for box_confidence, box_class_prob, box in zip(
                box_confidences, box_class_probs, boxes):
            # Compute the box scores for each output feature map
            box_scores_per_output = box_confidence * box_class_prob

            # For each individual box, keep the max of all the scores obtained
            max_box_scores = np.max(box_scores_per_output, axis=-1).reshape(-1)
            max_box_classes = np.argmax(
                box_scores_per_output, axis=-1).reshape(-1)

            box = box.reshape(-1, 4)

            # Filter out boxes based on the box score threshold
            filtering_mask = max_box_scores >= self.class_t
            filtered_box = box[filtering_mask]
            max_box_scores_filtered = max_box_scores[filtering_mask]
            max_box_classes_filtered = max_box_classes[filtering_mask]

            box_scores.append(max_box_scores_filtered)
            box_classes.append(max_box_classes_filtered)
            filtered_boxes.append(filtered_box)

        # Concatenate the results from all feature maps
        box_scores = np.concatenate(box_scores)
        box_classes = np.concatenate(box_classes)
        filtered_boxes = np.concatenate(filtered_boxes)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max suppression
            Args:
                filtered_boxes: numpy.ndarray of shape (?, 4) containing all of
                the filtered bounding boxes:
                    4: (x1, y1, x2, y2)
                box_classes: numpy.ndarray of shape (?,) containing the class
                number for the class that filtered_boxes predicts
                box_scores: numpy.ndarray of shape (?) containing the
                box scores
                for each box in filtered_boxes
            Returns: tuple of (box_predictions, predicted_box_classes,
            """
        # Initialize lists to hold the final predictions,
        # their classes, and scores
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Iterate over each unique class found in box_classes
        for box_class in np.unique(box_classes):
            # Find indices of all boxes belonging to the current class
            indices = np.where(box_classes == box_class)[0]

            # Extract subsets for the current class
            filtered_boxes_subset = filtered_boxes[indices]
            box_classes_subset = box_classes[indices]
            box_scores_subset = box_scores[indices]

            # Calculate the area of each box in the subset
            x1 = filtered_boxes_subset[:, 0]
            y1 = filtered_boxes_subset[:, 1]
            x2 = filtered_boxes_subset[:, 2]
            y2 = filtered_boxes_subset[:, 3]
            box_areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            # Sort boxes by their scores in descending order
            ranked = np.argsort(box_scores_subset)[::-1]

            # Initialize a list to keep track of boxes that
            # pass the suppression
            pick = []

            # Continue until all boxes are either picked or suppressed
            while ranked.size > 0:
                # Always pick the first box in the ranked list
                pick.append(ranked[0])

                # Compute the intersection over union (IOU) between
                # the picked box and all other boxes
                xx1 = np.maximum(x1[ranked[0]], x1[ranked[1:]])
                yy1 = np.maximum(y1[ranked[0]], y1[ranked[1:]])
                xx2 = np.minimum(x2[ranked[0]], x2[ranked[1:]])
                yy2 = np.minimum(y2[ranked[0]], y2[ranked[1:]])
                inter_areas = np.maximum(0, xx2 - xx1 + 1) * np.maximum(
                    0, yy2 - yy1 + 1)
                union_areas = box_areas[ranked[0]] + box_areas[
                    ranked[1:]] - inter_areas
                IOU = inter_areas / union_areas

                # Keep only boxes with IOU below the threshold
                updated_indices = np.where(IOU <= self.nms_t)[0]
                ranked = ranked[updated_indices + 1]

            # Update the final lists with the picks for this class
            pick = np.array(pick)
            box_predictions.append(filtered_boxes_subset[pick])
            predicted_box_classes.append(box_classes_subset[pick])
            predicted_box_scores.append(box_scores_subset[pick])

        # Concatenate the lists into final arrays
        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
