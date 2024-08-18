#!/usr/bin/env python3
"""depth of a decision tree"""
import numpy as np


class Node:
    """representing a node in a decision tree
    Attributes:
        feature: int representing the index of the feature to make a decision
        threshold: float threshold for the feature
        left_child: left node in the decision tree
        right_child: right node in the decision tree
        is_leaf: bool indicating if the node is a leaf
        is_root: bool indicating if the node is the root
        depth: depth of the node in the tree"""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """calculate the maximum depth below the current node"""
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(), self.right_child.max_depth_below()
            )

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below this node.

        Args:
        only_leaves: bool indicating if only leaves should be counted

        Returns:
            int representing the number of nodes below this node
        """
        if only_leaves:
            return self.left_child.count_nodes_below(
                only_leaves=True
            ) + self.right_child.count_nodes_below(only_leaves=True)
        else:
            return (
                1
                + self.left_child.count_nodes_below()
                + self.right_child.count_nodes_below()
            )

    def __str__(self):
        """string representation of the node"""
        if self.is_root:
            node_str = (
                "root [feature={self.feature}, "
                "threshold={self.threshold}]\n".format(self=self)
            )
        else:
            node_str = (
                "-> node [feature={self.feature}, "
                "threshold={self.threshold}]\n".format(self=self)
            )

        left_str = (
            self.left_child_add_prefix(self.left_child.__str__())
            if self.left_child
            else ""
        )
        right_str = (
            self.right_child_add_prefix(self.right_child.__str__())
            if self.right_child
            else ""
        )

        return node_str + left_str + right_str

    def left_child_add_prefix(self, text):
        """Add prefix to the left child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix to the right child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["    " + "   " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text


class Leaf(Node):
    """representing a leaf in a decision tree
    Attributes:
        value: value to be returned when the leaf is reached
        depth: depth of the node in the tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """calculate the maximum depth below the current node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Calculate the number of nodes below this node."""
        return 1

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """representing a decision tree
    Attributes:
        root: root node of the decision tree
        explanatory: numpy.ndarray of shape (m, n) containing the input data
        target: numpy.ndarray of shape (m,) containing the target data
        max_depth: int representing the maximum depth of the tree
        min_pop: int representing the minimum number of data points in a node
        seed: int for the random number generator
        split_criterion: string representing the type of split criterion
        predict: method to predict the value of a data point"""

    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None
    ):
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
        """calculate the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Calculate the number of nodes in the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return self.root.__str__()
