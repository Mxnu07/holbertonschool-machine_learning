#!/usr/bin/env python3
"""
This module implements a simple Decision Tree classifier.

Classes:
    Node: Represents a node in the decision tree,
      which can be either a decision node or a leaf.
    Leaf: Inherits from Node and represents a leaf in the decision tree,
      storing the predicted value.
    Decision_Tree: Implements a decision
    tree classifier that supports growing a tree, calculating
                   its depth, and making predictions.

Usage:
    This module allows for training a decision tree classifier on data,
      and retrieving information
    such as the depth of the tree and predictions.
"""
import numpy as np


class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth below this node."""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())


class Leaf(Node):
    """Represents a leaf node in the decision tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of this leaf node."""
        return self.depth


class Decision_Tree:
    """
    Implements a decision tree classifier.

    Attributes:
        max_depth (int): Maximum allowed depth of the tree.
        min_pop (int): Minimum population of data points
          required to split a node.
        rng (Generator): Random number generator instance.
        root (Node): The root node of the decision tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Calculates and returns the depth of the tree."""
        return self.root.max_depth_below()