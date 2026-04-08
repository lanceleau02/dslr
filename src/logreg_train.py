from src import sys, pd, np

def logreg_train(df):
    pass

def main():
    df = pd.read_csv("./data/dataset_train.csv")
    logreg_train(df)

if __name__ == "__main__":
    main()