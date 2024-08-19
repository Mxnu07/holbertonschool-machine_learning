#!/usr/bin/env python3
"""
Isolation_Random_Forest for outlier detection using
Isolation Trees.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """ This class defines an Isolation Random Forest. """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ Initializes the Isolation Random Forest. """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """ Makes predictions for a given set of examples. """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """ Fits the model to the training data. """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """ Returns the top n_suspects with the smallest depths. """
        # Calculate the mean depth for each data point using predict method
        depths = self.predict(explanatory)
        # Get the indices that would sort the depths array in ascending order
        sorted_indices = np.argsort(depths)
        # Select the top n suspects with the smallest depths
        suspect_data = explanatory[sorted_indices[:n_suspects]]
        suspect_depths = depths[sorted_indices[:n_suspects]]
        return suspect_data, suspect_depths
