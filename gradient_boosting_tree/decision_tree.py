"""This module defines the Node and DecisionTree classes."""
import numpy as np
import pandas as pd

class Node():
    """This class defines the attributes of a tree node."""
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree():
    """This class defines a decision tree regressor.
    
    Args:
        min_samples (int, optional): The minimum number of samples required to split an
        internal node. Defaults to 5.
        max_depth (int, optional): The maximum depth of the tree. If None, then nodes are
        expanded until all leaves are pure or until all leaves contain less than min_samples
        samples. Defaults to 3.
    """
    def __init__(self, min_samples: int = 5, max_depth: int = 3):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        # TODO: check type/shape of y
        """Builds a decision tree regressor from the training set (X, y).

        Args:
            X (pd.DataFrame): Array like dataframe of shape (n_samples, n_features).
            y (pd.DataFrame): Array like dataframe of shape (n_samples, ).
        """
        X = X.to_numpy()
        self.root = self._grow_tree(X, y)

    def predict(self, X: pd.DataFrame) -> list:
        """Predicts the y values based on the value of X.

        Args:
            X (pd.DataFrame): Array like dataframe of shape (n_samples, n_features).

        Returns:
            list: List of predictions for the y values.
        """
        # TODO: check type of return
        X = X.to_numpy()
        return [self._transverse(x, self.root) for x in X]

    def _transverse(self, x: list, node: Node) -> float:
        # TODO check type for x
        """_summary_

        Args:
            x (list): _description_
            node (Node): _description_

        Returns:
            float: The value of the leaf node when starting with the given x values.
        """
        if node.value is None:
            if x[node.feature] <= node.threshold:
                return self._transverse(x, node.left)
            return self._transverse(x, node.right)
        return node.value

    def _grow_tree(self, X: pd.DataFrame, y: pd.DataFrame, depth: int = 0) -> Node:
        n_samples = X.shape[0]
        if n_samples >= self.min_samples and depth <= self.max_depth:
            index, value = self._best_split(X, y)
            left_mask = X[:, index] <= value
            right_mask = X[:, index] > value

            left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
            right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

            return Node(feature = index, threshold = value, left = left, right = right)

        return Node(value = self._leaf_node(y))

    def _best_split(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple:
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            y (pd.DataFrame): _description_

        Returns:
            tuple: _description_
        """
        best_rss = float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature_i in range(n_features):
            for threshold in np.unique(X[:, feature_i]):
                # TODO: try to optimize <= or >=
                left_mask = X[:, feature_i] <= threshold
                right_mask = X[:, feature_i] > threshold
                rss = self._rss(y, left_mask, right_mask)
                if rss < best_rss:
                    best_rss = rss
                    best_feature = feature_i
                    best_threshold = threshold

        return best_feature, best_threshold

    def _rss(self, y: pd.DataFrame, left, right) -> float:
        # TODO: check left and right and return types
        """Calculates the Residual Sum of Squares (RSS).

        Args:
            y (pd.DataFrame): _description_
            left (_type_): _description_
            right (_type_): _description_

        Returns:
            float: The RSS for that data split.
        """
        y_left = y[left]
        y_right = y[right]
        rss_left = np.sum((y_left - np.mean(y_left)) ** 2)
        rss_right = np.sum((y_right - np.mean(y_right)) ** 2)
        return rss_left + rss_right

    def _leaf_node(self, y: pd.DataFrame) -> float:
        """Calculates the value of a leaf node.

        Args:
            y (pd.DataFrame): _description_

        Returns:
            float: The value of the leaf node.
        """
        return np.mean(y)
