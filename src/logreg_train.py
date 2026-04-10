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


def one_vs_all_training(df, X_matrix):
    houses = sorted(df['Hogwarts House'].dropna().unique())

    weights = {}

    for house in houses:
        y = np.where(df['Hogwarts House'] == house, 1, 0)
        theta = np.zeros(X_matrix.shape[1])
        optimized_theta, cost_history = gradient_descent(X_matrix, y, theta,
                                                         0.1, 1000)
        weights[house] = optimized_theta.tolist()

    return weights


def logreg_predict(df):
    X_matrix, mu, sigma, columns = data_preprocessing(df)
    weights_data = one_vs_all_training(df, X_matrix)

    json_data = {
        "features": columns,
        "mean":     mu.tolist(),
        "std":      sigma.tolist(),
        "weights":  weights_data,
    }

    with open("data/model.json", "w") as f:
        json.dump(json_data, f)


def main():
    df = open_file()
    logreg_predict(df)


if __name__ == "__main__":
    main()
