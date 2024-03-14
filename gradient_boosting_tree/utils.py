""""""
import numpy as np

def train_test_split(X, y, train_size):
    split_idx = len(y) - int(len(y) // (1 / train_size))
    X_test, X_train = X[:split_idx], X[split_idx:]
    y_test, y_train = X[:split_idx], X[split_idx:]
    return X_test, X_train, y_test, y_train

def performance_evaluation(y, y_pred):
    mae = np.mean(abs(y - y_pred))
    mse = np.square(np.subtract(y, y_pred)).mean()
    ssr = np.sum(np.square(np.subtract(y, y_pred)))
    sst = np.sum(np.square(np.subtract(y, np.mean(y))))
    r2 = 1 - ssr/sst
    
    print(f'Mean absolute error is {mae:.4f}')
    print(f'MSE (test): {mse:.4f}')
    print(f'RMSE (test): {mse**0.5:.4f}')
    print(f'R-squared is {r2:.4f}')
