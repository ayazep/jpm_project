import numpy as np

class Node():
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree():
    def __init__(self, min_samples = 5, max_depth = 3):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = X.to_numpy()
        self.root = self.grow_tree(X, y)

    def predict(self, X):
        X = X.to_numpy()
        return [self.transverse(x, self.root) for x in X]

    def transverse(self, x, tree):
        if tree.value is None:
            if x[tree.feature] <= tree.threshold:
                return self.transverse(x, tree.left)
            return self.transverse(x, tree.right)
        return tree.value

    def grow_tree(self, X, y, depth = 0):
        n_samples = X.shape[0]
        if n_samples >= self.min_samples and depth <= self.max_depth:
            index, value = self.best_split(X, y)
            left_mask = X[:, index] <= value
            right_mask = X[:, index] > value

            left = self.grow_tree(X[left_mask], y[left_mask], depth + 1)
            right = self.grow_tree(X[right_mask], y[right_mask], depth + 1)

            return Node(feature = index, threshold = value, left = left, right = right)

        return Node(value = self.leaf_node(y))

    def best_split(self, X, y):
        best_rss = float('inf')
        best_value = None
        best_index = None
        n_features = X.shape[1]

        for feature_i in range(n_features):
            for sample in np.unique(X[:, feature_i]):
                # TODO: try to optimize <= or >=
                left_mask = X[:, feature_i] <= sample
                right_mask = X[:, feature_i] > sample
                rss = self.rss(y, left_mask, right_mask)
                if rss < best_rss:
                    best_rss = rss
                    best_value = sample
                    best_index = feature_i

        return best_index, best_value

    def rss(self, y, left, right):
        y_left = y[left]
        y_right = y[right]
        rss_left = np.sum((y_left - np.mean(y_left)) ** 2)
        rss_right = np.sum((y_right - np.mean(y_right)) ** 2)
        return rss_left + rss_right

    def leaf_node(self, y):
        return np.mean(y)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature), "<=", tree.threshold)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
