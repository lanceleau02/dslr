from src import plt
from src.maths_utils import pearson_corr
from src.open_file import open_file
from src.utils import get_abbreviation
import json, sys
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    z = np.dot(X, theta)
    h = sigmoid(z)
    epsilon = 1e-15
    return -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(X)
    cost_history = []

    for i in range(iterations):
        # make a prediction
        H = sigmoid(X.dot(theta))
        # calculate the error
        error = H - y
        # calculate gradient descent
        X_t = X.transpose(1, 0)
        gradient = np.dot(((1 / m) * X_t), error)
        # update the weights
        theta = theta - alpha * gradient
        # call compute_cost() function
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

def data_preprocessing(df):
    # 2. Filter Features
    X = df[['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']]
    X = X.fillna(X.mean())

    # 3. Feature Scaling
    mu = X.mean()
    sigma = X.std()
    X_scaled = (X - mu) / sigma

    # 3.1 Saving mu and sigma to JSON file
    scaling_data = {
        "means": mu.to_dict(),
        "stds": sigma.to_dict()
    }

    with open('scaling_params.json', 'w') as f:
        json.dump(scaling_data, f, indent=4)

    # 4. Add the intercept
    X_matrix = X_scaled.to_numpy()
    ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((ones, X_matrix))

    return X_matrix

def one_vs_all_training(df, X_matrix):
    # isolate the targets
    houses = df['Hogwarts House'].dropna().unique()

    # create dict to store weigths
    weights = {}

    for house in houses:
        # temporary binary array
        y = np.where(df['Hogwarts House'] == house, 1, 0)
        # initialize a theta array of zeros
        theta = np.zeros(X_matrix.shape[1])
        # run gradient_descent() function
        optimized_theta, cost_history = gradient_descent(X_matrix, y, theta, 0.1, 1000)
        # store the optimized weights in the dict
        weights[house] = optimized_theta.tolist()

    weights_data = {
        "weights": weights
    }

    with open('weights_data.json', 'w') as f:
        json.dump(weights_data, f, indent=4)

def prediction():
    df = pd.read_csv("data/dataset_test.csv")

    with open('scaling_params.json', 'r') as file:
        scaling_data = json.load(file)

    with open('weights_data.json', 'r') as file:
        data = json.load(file)

    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
                'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
                'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    X_test = df[features]
    X_test = X_test.fillna(X_test.mean())

    saved_mu = pd.Series(scaling_data["means"])
    saved_sigma = pd.Series(scaling_data["stds"])
    
    X_test_scaled = (X_test - saved_mu) / saved_sigma

    X_matrix = X_test_scaled.to_numpy()
    ones = np.ones((X_matrix.shape[0], 1))
    X_matrix = np.hstack((ones, X_matrix))

    probabilities = {}
    
    for house, theta_list in data['weights'].items():
        theta = np.array(theta_list)
        probabilities[house] = sigmoid(X_matrix.dot(theta))

    prob_df = pd.DataFrame(probabilities)
    
    predictions = prob_df.idxmax(axis=1)

    result_df = pd.DataFrame({
        'Index': df['Index'],
        'Hogwarts House': predictions
    })

    result_df.to_csv('houses.csv', index=False)
    print("Success! Predictions saved to 'houses.csv'.")

def main():
    args = sys.argv[1:]
    filename = args[0]
    df = pd.read_csv(filename)

    X_matrix = data_preprocessing(df)
    one_vs_all_training(df, X_matrix)
    prediction()

if __name__ == "__main__":
    main()