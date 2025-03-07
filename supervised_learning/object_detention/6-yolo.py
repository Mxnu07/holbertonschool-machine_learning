#!/usr/bin/env python3
"""This module contains the Yolo class
that uses the Yolo v3 algorithm to perform object detection
includes processing output method
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.activations import sigmoid  # type: ignore
import cv2
import os


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

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder
        Args:
            folder_path: a string representing the path to the folder holding
                all the images to load
        Returns: a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """

        images = []
        images_paths = []
        # Load the images
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            if image_path is not None:
                images_paths.append(image_path)
                image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
        return (images, images_paths)

    def preprocess_images(self, images):
        """
        Preprocess a list of images for model input.

        This function resizes and rescales the images to the input
        dimensions expected by the model,
        and normalizes their pixel values to the range [0, 1].

        Parameters:
        - images (list of np.ndarray): List of images to preprocess.
        Each image is represented as a numpy array.

        Returns:
        - tuple: A tuple containing two elements:
            - np.ndarray: An array of preprocessed images ready for
            model input.
            - np.ndarray: An array of the original shapes of the
            images before preprocessing.
        """

        # Initialize lists to store preprocessed images
        # and their original shapes
        pimages = []
        image_shapes = []

        # Retrieve input dimensions from the model
        # TensorFlow 1.x syntax (for compatibility)
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        # Loop through each image in the input list
        for image in images:
            # Append the original shape (height, width) of the
            # image to image_shapes list
            image_shapes.append(np.array([image.shape[0], image.shape[1]]))
            # Resize the image to the input dimensions of the
            # model using cubic interpolation
            pimage = cv2.resize(image, (
                input_width, input_height), interpolation=cv2.INTER_CUBIC)
            # Rescale the pixel values of the image to the range [0, 1]
            pimage = pimage / 255.0
            # Append the preprocessed image to the pimages list
            pimages.append(pimage)

        # Convert the lists of preprocessed images and their
        # original shapes to numpy arrays
        image_shapes = np.array(image_shapes)
        pimages = np.array(pimages)

        # Return the preprocessed images and their original shapes
        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays an image with all boundary boxes, class names, and box scores.

        This function draws rectangles around detected objects in an image, annotates them with the
        class names and the confidence scores, and displays the image. The user can save the annotated
        image by pressing 's'.

        Parameters:
        - image (np.ndarray): The image on which to draw the boxes.
        - boxes (np.ndarray): Array of bounding box coordinates (x1, y1, x2, y2) for each detection.
        - box_classes (np.ndarray): Array of class indices for each detected object.
        - box_scores (np.ndarray): Array of scores for each detection.
        - file_name (str): The name of the file where the image with drawn boxes will be saved if 's' is pressed.

        Note:
        - The function assumes the use of OpenCV for image manipulation and display.
        - The color format for the drawn rectangles and text is BGR.
        """

        # Iterate through all detected boxes
        for i, box in enumerate(boxes):
            # Extract the coordinates of the current box
            x_1, y_1, x_2, y_2 = box

            # Draw the rectangle around the detected object
            start_point = (int(x_1), int(y_1))  # Top-left corner
            end_point = (int(x_2), int(y_2))    # Bottom-right corner
            color = (255, 0, 0)  # Blue color in BGR
            thickness = 2  # Thickness of the rectangle border
            cv2.rectangle(image, start_point, end_point, color, thickness)

            # Prepare the text to be displayed (class name and score)
            text = "{} {:.2f}".format(self.class_names[box_classes[i]], box_scores[i])
            org = (int(x_1), int(y_1) - 5)  # Position of the text
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font style
            fontScale = 0.5  # Font scale
            color = (0, 0, 255)  # Red color in BGR for the text
            thickness = 1  # Thickness of the text
            lineType = cv2.LINE_AA  # Type of the line used for the text
            cv2.putText(image, text, org, font, fontScale, color, thickness, lineType)

        # Display the image with drawn boxes and annotations
        cv2.imshow(file_name, image)

        # Wait for the 's' key to be pressed to save the image
        key = cv2.waitKey(0)
        if key == ord('s'):
            # Create the 'detections' directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')
            # Save the image with annotations
            cv2.imwrite('detections/' + file_name, image)

        # Close all OpenCV windows
        cv2.destroyAllWindows()
