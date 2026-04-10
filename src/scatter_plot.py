from matplotlib.widgets import RadioButtons

from src import plt
from src.maths_utils import pearson_corr
from src.utils import get_abbreviation, open_file


def scatter_plot(df):
    """
    Generates scatter plots of numeric features from the input DataFrame,
    displaying the relationships between the features for different
    categories of "Hogwarts House".

    :param df: The input DataFrame containing numeric data columns and a
    categorical column "Hogwarts House".
    :type df: pandas.DataFrame
    :return: None
    """
    # Filter for numeric columns and ignore index
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(
        columns=['Index'], errors='ignore')
    features = list(numeric_df.columns)

    if not features:
        print("No features available for scatter plot.")
        return

    feature_labels = [get_abbreviation(f) for f in features]
    label_to_feature = dict(zip(feature_labels, features))

    num_features = len(features)

    # Configure grid layout: 4 columns and rows based on feature count
    num_plots = num_features - 1
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes_grid = plt.subplots(nrows=num_rows, ncols=num_cols,
                                  figsize=(20, 4 * num_rows))
    plt.subplots_adjust(
        left=0.18,  # keep space for radio buttons
        right=0.99,  # remove right blank space
        top=0.95,  # remove top blank space
        bottom=0.05,  # remove bottom blank space
        hspace=0.4,
        wspace=0.3
    )
    axes = axes_grid.flatten()

    houses = df['Hogwarts House'].unique()
    colors = {'Gryffindor': 'red', 'Slytherin': 'green',
              'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

    # Radio buttons for interactive feature selection
    rax = plt.axes([0.01, 0.25, 0.12, 0.5], facecolor='#f0f0f0')
    radio = RadioButtons(rax, feature_labels)

    # Select default feature to show (the most correlated)
    best_f1, best_f2 = find_best_pair(df)
    default_label = get_abbreviation(best_f1) if best_f1 else feature_labels[0]

    def update(label):
        """
        Updates scatter plots for features of different Hogwarts houses.

        :param label: The feature name selected to update the scatter plots.
        :type label: str
        :return: None
        """
        feature1 = label_to_feature[label]
        other_features = [f for f in features if f != feature1]

        # Iterate through subplot axes to update each scatter plot
        for i, ax in enumerate(axes):
            if i < len(other_features):
                ax.clear()
                feature2 = other_features[i]

                # Calculate correlation on data for these features
                subset = df[[feature1, feature2]].dropna()
                if len(subset) > 1:
                    correlation = pearson_corr(subset[feature1].tolist(),
                                               subset[feature2].tolist())
                else:
                    correlation = 0.0

                # Plot data for each house with its assigned color
                for house in houses:
                    house_data = df[df['Hogwarts House'] == house]
                    ax.scatter(house_data[feature1], house_data[feature2],
                               alpha=0.5, s=5, label=house,
                               color=colors[house])

                # Display the Pearson correlation coefficient calculated on
                # subset
                ax.set_title(
                    f'{feature1} vs {feature2}\nCorr: {correlation:.4f}',
                    fontsize=8)
                ax.set_xlabel('', fontsize=7)
                ax.set_ylabel('', fontsize=7)
                ax.grid(True)
                ax.set_visible(True)
            else:
                ax.set_visible(False)

        plt.draw()

    radio.on_clicked(update)

    update(default_label)

    plt.show()


def find_best_pair(df):
    """
    Finds the pair of features with the highest absolute Pearson correlation.
    """
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(
        columns=['Index'], errors='ignore')
    features = list(numeric_df.columns)
    if not features:
        return None, None

    best_pair = None
    best_abs = -1.0

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            r = pearson_corr(df[f1].tolist(), df[f2].tolist())
            ar = abs(r)
            if ar > best_abs:
                best_abs = ar
                best_pair = (f1, f2)
    return best_pair


def calculate_correlation(df):
    """
    Calculate the pair of features with the highest absolute Pearson
    correlation coefficient from the given DataFrame, ignoring the 'Index'
    column.

    :param df: The input pandas DataFrame containing data for analysis.
               It must include numeric columns to compute correlations.
    :type df: pandas.DataFrame
    :return: None
    """
    # Filter numeric columns and drop unnecessary ones
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(
        columns=['Index'])
    features = list(numeric_df.columns)

    if not features:
        print("Most similar pair: not found (no features)")
        return
    best_pair = None
    best_abs = -1.0
    best_val = 0.0

    # Iterates over unique feature pairs to find the most correlated pair
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            r = pearson_corr(df[f1].tolist(), df[f2].tolist())
            ar = abs(r)
            # Find the best correlation (closest to 1)
            if best_abs < ar:
                best_abs = ar
                best_val = r
                best_pair = (f1, f2)

    if best_pair is not None:
        print(
            f"Most similar pair: {best_pair} with correlation {best_val:.2f}")
    else:
        print("Most similar pair: not found")


def main():
    """
    Main entry point of the script. Loads data, displays the interactive
    scatter plot, and calculates feature correlations.
    """
    df = open_file(drop_na=True)
    scatter_plot(df)
    calculate_correlation(df)


if __name__ == "__main__":
    main()
