import json

import numpy as np

from src.maths_utils import sigmoid
from src.open_file import open_file


def train_logreg(X, y, learning_rate=0.01, epochs=1000):
    """
    Trains a logistic regression model using batch gradient descent.

    This function performs logistic regression training by iteratively updating
    the weight vector (`theta`) using gradient descent with the binary
    cross-entropy
    loss function.

    :param X: The feature matrix of shape (m, n), where `m` is the number of
    samples
        and `n` is the number of features.
    :param y: The target vector of shape (m,), where `m` is the number of
    samples.
        It should contain binary labels (0 or 1) corresponding to each
        sample in X.
    :param learning_rate: The learning rate for gradient descent. Defaults
    to 0.01.
    :param epochs: The total number of iterations (epochs) over the training
    data.
        Defaults to 1000.
    :return: The optimized weight vector (`theta`) of shape (n,), where `n`
    is the
        number of features in the input feature matrix X.
    :rtype: numpy.ndarray
    """
    m, n = X.shape
    theta = np.zeros(n)

    # Batch Gradient Descent
    for _ in range(epochs):
        z = X @ theta
        h = sigmoid(z)

        # Gradient calculation
        gradient = (1 / m) * (X.T @ (h - y))
        theta -= learning_rate * gradient

    return theta


def main() -> None:
    """
    Main function for training a logistic regression model to classify
    Hogwarts House
    using a dataset. The function performs data preprocessing, training,
    and model
    saving in JSON format. Preprocessing includes dropping irrelevant columns,
    imputing missing values, feature scaling, and adding a bias term. Logistic
    regression is implemented with a one-vs-rest strategy for multiclass
    classification.

    :return: None
    """

    # Load data
    data_file = open_file()

    # Target variable
    y = data_file["Hogwarts House"]

    # Drop irrelevant columns
    X = data_file.drop(
        columns=["Index", "Hogwarts House", "First Name", "Last Name",
                 "Birthday", "Best Hand"])

    # Simple imputation
    X = X.fillna(X.mean())

    # Save stats for future predictions
    x_mean = X.mean()
    x_std = X.std()

    # Feature Scaling
    X = (X - x_mean) / x_std

    # Add bias
    X_values = X.values
    X_with_bias = np.c_[np.ones(X_values.shape[0]), X_values]

    houses = sorted(y.unique())
    thetas = {}

    # Train one-vs-rest
    for house in houses:
        y_binary = (y == house).astype(int).values
        theta = train_logreg(X_with_bias, y_binary)
        thetas[house] = theta

    json_data = {
        "features": X.columns.tolist(),
        "mean":     x_mean.tolist(),
        "std":      x_std.tolist(),
        "theta":    {house: theta.tolist() for house, theta in thetas.items()},
    }

    with open("model.json", "w") as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    main()
