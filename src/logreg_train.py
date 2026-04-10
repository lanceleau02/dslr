import json

import numpy as np

from src.maths_utils import sigmoid, compute_cost
from src.utils import open_file


def data_preprocessing(df):
    X = df.drop(
        columns=["Index", "Hogwarts House", "First Name", "Last Name",
                 "Birthday", "Best Hand"])
    X = X.fillna(X.mean())

    mu = X.mean()
    sigma = X.std()
    X_scaled = (X - mu) / sigma

    X_matrix = X_scaled.to_numpy()
    ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((ones, X_matrix))

    return X_matrix, mu, sigma, X.columns.tolist()


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(X)
    cost_history = []

    for _ in range(iterations):
        H = sigmoid(X.dot(theta))
        error = H - y
        X_t = X.transpose(1, 0)
        gradient = np.dot(((1 / m) * X_t), error)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


def mini_batch_gradient_descent(X, y, theta, alpha, iterations, batch_size=32):
    m = len(X)
    cost_history = []

    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            batch_m = len(X_batch)
            H = sigmoid(X_batch.dot(theta))
            error = H - y_batch
            X_t = X_batch.transpose(1, 0)
            gradient = np.dot(((1 / batch_m) * X_t), error)
            theta -= alpha * gradient

        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


def stochastic_gradient_descent(X, y, theta, alpha, iterations):
    m = len(X)
    cost_history = []

    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):
            xi = X_shuffled[i, :]
            yi = y_shuffled[i]
            hi = sigmoid(np.dot(xi, theta))
            error = hi - yi
            gradient = xi * error
            theta -= alpha * gradient

        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


def one_vs_all_training(df, X_matrix, method='batch'):
    houses = sorted(df['Hogwarts House'].dropna().unique())

    weights = {}

    for house in houses:
        y = np.where(df['Hogwarts House'] == house, 1, 0)
        theta = np.zeros(X_matrix.shape[1])
        if method == 'stochastic':
            optimized_theta, cost_history = stochastic_gradient_descent(
                X_matrix, y, theta, 0.01, 100)
        elif method == 'minibatch':
            optimized_theta, cost_history = mini_batch_gradient_descent(
                X_matrix, y,
                theta,
                0.1, 1000)
        else:
            optimized_theta, cost_history = gradient_descent(X_matrix, y,
                                                             theta,
                                                             0.1, 1000)
        weights[house] = optimized_theta.tolist()

    return weights


def logreg_train(df, method='batch'):
    X_matrix, mu, sigma, columns = data_preprocessing(df)
    weights_data = one_vs_all_training(df, X_matrix, method=method)

    json_data = {
        "features": columns,
        "mean":     mu.tolist(),
        "std":      sigma.tolist(),
        "weights":  weights_data,
    }

    with open("data/model.json", "w") as f:
        json.dump(json_data, f)


def main():
    import sys
    method = 'batch'
    filename = None

    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    df = open_file(filename=filename)
    logreg_train(df, method=method)


if __name__ == "__main__":
    main()
