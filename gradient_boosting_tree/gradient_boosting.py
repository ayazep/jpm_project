"""
Custom Gradient Boosting Regressor

This module implements a custom gradient boosting regressor for predicting continuous target
values. It utilizes decision trees as base estimators and iteratively improves predictions
by minimizing the residuals.

Classes:
    - CustomGradientBoostingRegressor: Defines a gradient boosting regressor with customizable
    parameters such as maximum tree depth, minimum samples for node split, learning rate,
    and number of estimators.

The gradient boosting regressor implemented here works by sequentially adding decision trees
to the ensemble, with each tree trained to correct the errors made by the previous trees.
Predictions are made by aggregating the predictions of all the individual trees with scaled weights.
"""
import numpy as np
import pandas as pd
from gradient_boosting_tree.decision_tree import DecisionTree

class CustomGradientBoostingRegressor:
    """Define a gradient boosting regressor.

    Parameters:
        max_depth (int, optional): The maximum depth of the individual decision trees.
        Defaults to 3.
        min_samples (int, optional): The minimum number of samples required to split an internal
        node in each decision tree. Defaults to 5.
        learning_rate (float, optional): The step size shrinkage used in updating the predictions
        at each iteration. Defaults to 0.1.
        n_estimators (int, optional): The number of boosting stages (the number of decision trees).
        Defaults to 50.

    Returns:
        None
    """
    def __init__(self, max_depth: int = 3, min_samples: int = 5, learning_rate: float = 0.1, 
                 n_estimators: int = 50):
        """Initialize GradientBoostingRegressor object with specified parameters."""
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

        # Initialize list to hold DecisionTree models
        self.trees = []
        # Create decision trees as base estimators
        for _ in range(n_estimators):
            tree = DecisionTree(min_samples=self.min_samples, max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Build a gradient boosting regressor from the training set (X, y).

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values corresponding to X.
        """
        # Initialize predictions with actual y values
        y_pred = np.copy(y)
        # Iterate over each decision tree
        for tree in self.trees:
            # Fit the tree to the data and residuals
            tree.fit(X, y_pred)
            # Make predictions using the current tree
            new_y_pred = tree.predict(X)
            # update predictions with scaled residuals
            y_pred -= np.multiply(self.learning_rate, new_y_pred)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the y values.

        Args:
            X (pd.DataFrame): The input samples for prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        # Initialize predictions with zeros
        y_pred = np.zeros(X.shape[0])
        # Iterate over each decision tree
        for tree in self.trees:
            # Make predictions using the current tree
            new_y_pred = tree.predict(X)
            # Update predictions by adding scaled predictions
            y_pred += np.multiply(self.learning_rate, new_y_pred)
        return y_pred
