"""
Decision Tree Regressor

This module implements a decision tree regressor for predicting continuous target values. It
includes classes for representing nodes in the decision tree and the decision tree regressor itself.

Classes:
    - Node: Represents a node in a decision tree, containing information about splitting criteria
    and predicted values.
    - DecisionTree: Defines a decision tree regressor, including methods for fitting the model
    and making predictions.

The decision tree regressor implemented here builds a binary tree recursively, splitting the dataset
into smaller subsets until a stopping criterion is met (minimum number of samples or maximum tree
depth). Predictions are made by traversing the tree based on the input features of the data points.
"""
import numpy as np
import pandas as pd

class Node:
    """Represent a node in a decision tree.

    Parameters:
        feature (int, optional): The index of the feature used for splitting at this node.
        Defaults to None.
        threshold (float, optional): The threshold value for splitting the feature.
        Defaults to None.
        left (Node, optional): The left child node. Defaults to None.
        right (Node, optional): The right child node. Defaults to None.
        value (float, optional): The predicted value if this node is a leaf. Defaults to None.
    """
    def __init__(self, feature: int = None, threshold: float = None,
                 left: 'Node' = None, right: 'Node' = None, value: float = None):
        """Initialize Node object with specified prarmeters."""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree():
    """Define a decision tree regressor.
    
    Parameters:
        min_samples (int, optional): The minimum number of samples required to split an
        internal node. Defaults to 5.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 3.
    """
    def __init__(self, min_samples: int = 5, max_depth: int = 3):
        """Initialize DecisionTree object with specified prarmeters."""
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def _mean_squared_error(self, y: np.ndarray) -> float:
        """Calculate the Mean Sqaured Error (MSE).

        Args:
            y (np.ndarray): The target values.

        Returns:
            float: The mean squared error.
        """
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _mse(self, y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> float:
        """Calculate the total Mean Squared Error (MSE) for the right and left nodes.

        Args:
            y (np.ndarray): The target values.
            left_mask (np.ndarray): Boolean mask indicating the samples in the left node.
            right_mask (np.ndarray): Boolean mask indicating the samples in the right node.

        Returns:
            float: The total mean squared error.
        """
        left = y[left_mask]
        right = y[right_mask]
        mse_left = self._mean_squared_error(left)
        mse_right = self._mean_squared_error(right)

        total_samples = len(left) + len(right)
        total_left = (len(left) / total_samples) * mse_left
        total_right = (len(right) / total_samples) * mse_right

        return total_left + total_right

    def _leaf_node(self, y: np.ndarray) -> float:
        """Calculate the value of a leaf node.

        Args:
            y (np.ndarray): The target values.

        Returns:
            float: The mean value of the target values.
        """
        return np.mean(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best feature index and threshold to split the node on.
        
        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Returns:
            tuple: A tuple containing the best feature index and threshold.
        """
        # Initialize variables to track the best split
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        # Iterate over features to find the best split
        for feature_i in range(n_features):
            for threshold in np.unique(X[:, feature_i]):
                left_mask = X[:, feature_i] <= threshold
                right_mask = X[:, feature_i] > threshold
                mse = self._mse(y, left_mask, right_mask)
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_i
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split the data into leaf nodes.
        
        Args:
            X (np.array): The input features.
            y (np.array): The target values.
            depth (int, optional): The current depth of the tree. Defaults to 0.

        Returns:
            Node: The root node of the grown tree.
        """
        # Check if further splitting is necessary
        n_samples = X.shape[0]
        if n_samples >= self.min_samples and depth <= self.max_depth:
            # Find the best split for the current node
            index, threshold = self._best_split(X, y)
            left_mask = X[:, index] <= threshold
            right_mask = X[:, index] > threshold

            # Recursively grow the left and right subtrees
            left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
            right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

            # Define a parent node with the best split
            return Node(feature = index, threshold = threshold, left = left, right = right)

        # Define a leaf node with the mean value of the target variable
        return Node(value = self._leaf_node(y))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Build a decision tree regressor from the training set (X, y).
        
        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values corresponding to X.
        """
        # Convert input DataFrame to numpy array
        X = X.to_numpy()
        # Grow the decision tree
        self.root = self._grow_tree(X, y)

    def _traverse(self, x: np.ndarray, node: Node) -> float:
        """Traverse the decision tree to find the predicted value for input data point.

        Args:
            x (np.ndarray): The input data point.
            node (Node): The current node being evaluated.

        Returns:
            float: The predicted value for the input data point.
        """
        if node.value is None:
            if x[node.feature] <= node.threshold:
                return self._traverse(x, node.left)
            return self._traverse(x, node.right)
        return node.value

    def predict(self, X: pd.DataFrame) -> list:
        """Predict the y values.

        Args:
            X (pd.DataFrame): The input samples for prediction.

        Returns:
            list: A list of predicted target values.
        """
        # Convert input DataFrame to numpy array
        X = X.to_numpy()
        # Predict the target values for the input data
        return [self._traverse(x, self.root) for x in X]
