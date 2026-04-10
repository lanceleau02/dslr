import json

import numpy as np
import pandas as pd

from src.maths_utils import sigmoid
from src.open_file import open_file


def predict(X, thetas):
    """
    Predicts the most probable class for each input sample based on the
    logistic regression
    probabilities computed for each class.

    :param X: A list or iterable of input samples. Each element in the
    iterable represents
              an input sample, where the sample is expected to be a feature
              vector.
    :param thetas: A dictionary mapping each class to its corresponding
    logistic regression
                   parameter vector. Each parameter vector is used to
                   compute the probability
                   for the associated class.

    :return: A list of predicted classes corresponding to the most probable
    class for each
             input sample. Each element in the returned list corresponds to
             the predicted
             label for the respective input sample.
    """
    predictions = []

    for x in X:
        probs = {}
        for house, theta in thetas.items():
            probs[house] = sigmoid(x @ theta)

        predictions.append(max(probs, key=probs.get))

    return predictions


def main() -> None:
    """
    Main entry point of the program. This function performs the following
    operations:

    1. Opens a data file using the `open_file` function.
    2. Loads a pre-trained model from `model.json`.
    3. Extracts necessary training-related details, such as features, means,
       standard deviations, and weights (theta values).
    4. Processes the test data by imputing missing values, normalizing the
    data,
       and adding a bias term to match the format used during training.
    5. Predicts results using the pre-trained model through the `predict`
       function.
    6. Outputs results in a CSV file named `houses.csv`.

    :return: None
    """
    data_file = open_file()

    with open("model.json", "r") as f:
        model = json.load(f)

    # Extract training-related details
    features = model["features"]
    x_mean = np.array(model["mean"])
    x_std = np.array(model["std"])
    thetas = {
        house: np.array(theta)
        for house, theta in model["theta"].items()
    }

    # Extract features used during training
    X_test = data_file[features]

    # Impute missing values using training means
    X_test = X_test.fillna(pd.Series(x_mean, index=features))

    # Normalize using training stats
    X_test = (X_test - x_mean) / x_std

    # Add bias
    X_values = X_test.values
    X_with_bias = np.c_[np.ones(X_values.shape[0]), X_values]

    preds = predict(X_with_bias, thetas)

    result = pd.DataFrame({
        "Index":          data_file["Index"],
        "Hogwarts House": preds
    })

    result.to_csv("houses.csv", index=False)


if __name__ == '__main__':
    main()
