import numpy as np
from gradient_boosting_tree.decision_tree import DecisionTree
import matplotlib.pyplot as plt
class GradientBoostingRegressor():
    def __init__(self, max_depth = 3, min_samples = 5, learning_rate = 0.1, n_estimators = 50):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTree(min_samples = self.min_samples, max_depth = self.max_depth)
            self.trees.append(tree)

    def fit(self, X, y):
        y_pred = np.copy(y)
        for tree in self.trees:
            tree.fit(X, y_pred)
            new_y_pred = tree.predict(X)
            y_pred = y_pred - np.multiply(self.learning_rate, new_y_pred)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            new_y_pred = tree.predict(X) # This is predicting the tree, not a recursive function
            y_pred += np.multiply(self.learning_rate, new_y_pred)
        return y_pred
