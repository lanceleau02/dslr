import math

import matplotlib.pyplot as plt

from src.utils import open_file


def plot_histogram(df, feature, ax):
    houses = df['Hogwarts House'].dropna().unique()

    for house in houses:
        house_data = df[df['Hogwarts House'] == house][feature].dropna()
        ax.hist(house_data, bins=25, alpha=0.5, label=house)

    ax.set_title(feature, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)


def main():
    df = open_file(drop_na=True)
    features = [col for col in df.columns if
                df[col].dtype in ['float64', 'int64'] and col != 'Index']

    if not features:
        print("No features available for histogram plotting.")
        return

    cols = 4
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12), squeeze=False)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        plot_histogram(df, feature, axes[i])

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=12,
               title="Hogwarts House")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    main()
