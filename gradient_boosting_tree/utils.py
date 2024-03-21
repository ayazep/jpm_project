"""
Performance Metrics and Utility Functions

This module provides functions for calculating performance metrics such as Mean Absolute Error,
Mean Squared Error, and R-Squared. It also includes utility functions for data splitting,
k-fold cross-validation, and grid search over hyperparameters.

Functions:
    - mae_val: Calculate Mean Absolute Error (MAE).
    - mse_val: Calculate Mean Square Error (MSE).
    - r_square: Calculate R-Squared.
    - performance_evaluation: Print all performance metrics (MAE, MSE, RMSE, R-squared).
    - train_test: Split the dataset into train and test sets.
    - k_fold: Perform k-fold cross-validation.
    - grid_search: Perform grid search over a set of hyperparameters.
"""
import itertools
import numpy as np
import pandas as pd
from gradient_boosting_tree.gradient_boosting import CustomGradientBoostingRegressor

def mae_val(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    Args:
        y (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The Mean Absolute Error.
    """
    return np.mean(abs(y - y_pred))

def mse_val(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Square Error.

    Args:
        y (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The Mean Square Error.
    """
    return np.mean(np.square(np.subtract(y, y_pred)))

def r_square(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-Squared.

    Args:
        y (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.

    Returns:
        float: The R-Squared value.
    """
    ssr = np.sum(np.square(np.subtract(y, y_pred)))
    sst = np.sum(np.square(np.subtract(y, np.mean(y))))
    return 1 - ssr/sst

def performance_evaluation(y: np.ndarray, y_pred: np.ndarray):
    """Print all the performance metrics.

    Args:
        y (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
    """
    mae = mae_val(y, y_pred)
    mse = mse_val(y, y_pred)
    r2 = r_square(y, y_pred)

    print(f'Mean Absolute Error (MAE) is {mae:.4f}')
    print(f'Mean Squared Error (MSE) is {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE) is {mse**0.5:.4f}')
    print(f'R-squared is {r2:.4f}')


def train_test(X: pd.DataFrame, y: pd.Series, train_size: float = 0.8, seed: int = None):
    """Split the dataset into train and tests sets.

    Args:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target values.
        train_size (float, optional): The proportion of the dataset to include in the
        train split. Defaults to 0.8.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing train-test split of input features and target values.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Get number of samples
    num_samples = len(X)
    # Create shuffled indices
    index = np.arange(num_samples)
    np.random.shuffle(index)

    # Shuffle X and y using shuffled indices
    X = X.loc[index]
    y = y.loc[index]

    # Detemine split inde based on train size
    split_i = int(num_samples * train_size)
    # Split dataset into train and test sets
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def k_fold(model, k_X_train: pd.DataFrame, k_y_train: pd.Series, n_folds: int, seed: int = None):
    """Perform k-fold cross-validation.

    Args:
        model: The model to be evaluated.
        k_X_train (pd.DataFrame): The input features for training.
        k_y_train (pd.Series): The target values for training.
        n_folds (int): Number of folds for cross-validation.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the average R-squared score, true target values, and predicted
        target values.
    """
    # Reset indices
    k_X_train = k_X_train.reset_index(drop = True)
    k_y_train = k_y_train.reset_index(drop = True)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Create shuffled indices
    index = list(np.arange(len(k_X_train)))
    np.random.shuffle(index)

    # Initialize lists to store results
    results = []
    predictions = []
    y_tests = []

    # Iterate through each fold
    for i in range(n_folds):
        # Determine testing and training indices
        marker = int(len(k_X_train) / n_folds)
        testing_index = index[marker * i: marker * (i + 1)]
        training_index = index[:marker * i] + index[marker * (i + 1):]

        # Handle the last fold
        if i == (n_folds - 1):
            testing_index += index[marker * (i + 1):]
            training_index = training_index[:marker * i]

        # Get training and testing data
        X_train = k_X_train.loc[training_index]
        X_test = k_X_train.loc[testing_index]
        y_train = k_y_train.loc[training_index]
        y_test = k_y_train.loc[testing_index]

        # Fit the model and make predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate and store R^2 score
        r2 = r_square(y_test, y_pred)
        results.append(r2)

        # Store predictions and true labels
        predictions.append(y_pred)
        y_tests.append(y_test)

    # Calculate average R^2 score
    avg_r2 = np.mean(results)

    return avg_r2, y_tests, predictions


def grid_search(parameters: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                y_test: pd.Series, n_folds: int = 0, seed: int = None):
    """Perform grid search over a set of hyperparameters.
    
    Args:
        parameters (dict): A dictionary containing hyperparameters to be searched.
        X_train (pd.DataFrame): The input features for training.
        X_test (pd.DataFrame): The input features for testing.
        y_train (pd.Series): The target values for training.
        y_test (pd.Series): The target values for testing.
        n_folds (int, optional): Number of folds for cross-validation. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the best R-squared score, the best combination of hyperparameters,
        predicted target values corresponding to the best combination, and true target values.
    """
    # Initialize variables
    count = 0
    best_r2 = 0
    best_y_pred = []
    best_y_test = []
    best_combo = 0

    # Generate combinations of hyperparameters
    param_combinations = list(itertools.product(*parameters.values()))

    # Iterate over each combination
    for combo in param_combinations:
        count += 1
        print(f'combination: {count}', end='\r')

        # Create model with current combination of hyperparameters
        model = CustomGradientBoostingRegressor(**dict(zip(parameters.keys(), combo)))

        # Perform k-fold cross-validation or use a single train-test split
        y_pred = None
        r2 = 0
        if n_folds > 0:
            r2, y_test, y_pred = k_fold(model, X_train, y_train, n_folds, seed)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r_square(y_test, y_pred)

        # Update best results if current combination improves upon previous best
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_y_pred = y_pred
            best_y_test = y_test

    return best_r2, best_combo, best_y_pred, best_y_test
