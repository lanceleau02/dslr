import json

import numpy as np
import pandas as pd

from src.maths_utils import sigmoid
from src.utils import print_error, open_file


def preprocess_data(df, model):
    """
    Remplit les valeurs manquantes et normalise les données à partir des
    paramètres du modèle.
    """
    features = model["features"]
    mu = np.array(model["mean"])
    sigma = np.array(model["std"])

    X = df[features]
    X = X.fillna(X.mean())

    X_scaled = (X - mu) / sigma

    X_matrix = X_scaled.to_numpy()
    ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((ones, X_matrix))

    return X_matrix


def compute_predictions(X_matrix, weights):
    """
    Calcule les probabilités pour chaque maison et renvoie la maison avec la
    probabilité max.
    """
    probabilities = {}

    for house, theta_list in weights.items():
        theta = np.array(theta_list)
        probabilities[house] = sigmoid(X_matrix.dot(theta))

    prob_df = pd.DataFrame(probabilities)
    return prob_df.idxmax(axis=1)


def save_predictions(df, predictions, filename='data/houses.csv'):
    """
    Sauvegarde les prédictions dans un fichier CSV.
    """
    result_df = pd.DataFrame({
        'Index':          df['Index'],
        'Hogwarts House': predictions
    })

    result_df.to_csv(filename, index=False)
    print(f"Success! Predictions saved to '{filename}'.")


def logreg_predict(df, model):
    weights = {
        house: np.array(theta)
        for house, theta in model["weights"].items()
    }

    X_matrix = preprocess_data(df, model)
    predictions = compute_predictions(X_matrix, weights)
    save_predictions(df, predictions)


def main():
    df = open_file()
    try:
        with open("data/model.json", "r") as f:
            model = json.load(f)
    except FileNotFoundError:
        print_error("Model file not found.")
    logreg_predict(df, model)


if __name__ == "__main__":
    main()
