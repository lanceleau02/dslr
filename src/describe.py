from src.arg_checker import check_args
from src import pd, np
from src.maths_utils import sum_, sort_, mean_, std_, min_, max_, percentile_
from src.open_file import open_file


def describe(dataset):
    """
    Analyzes the dataset to compute basic statistical metrics for numeric
    columns, ignoring any non-numeric columns or missing values.

    :param dataset: A dataset input.
    :return: None.
    """
    df = dataset.loc[:, dataset.columns != 'Index']
    stats = {}
    for col in df.columns:

        # If the column is not numeric, skip it
        if df[col].dtype not in [np.float64, np.int64]:
            continue

        data = df[col].dropna().values

        n = sum_(1 for _ in data)
        if n == 0:
            continue

        sorted_data = sort_(data)

        stats[col] = {
            "Count": n,
            "Mean":  mean_(data),
            "Std":   std_(data),
            "Min":   min_(data),
            "25%":   percentile_(sorted_data, 0.25),
            "50%":   percentile_(sorted_data, 0.50),
            "75%":   percentile_(sorted_data, 0.75),
            "Max":   max_(data)
        }
    print(pd.DataFrame(stats))


def main():
    """
    Main entry point of the script that processes command-line arguments
    and describes a dataset.
    """
    dataset = open_file()
    describe(dataset)


if __name__ == "__main__":
    main()
