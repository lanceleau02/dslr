import matplotlib.pyplot as plt
import numpy as np

from src.open_file import open_file


def pair_plot(data):
    house_colors = {
        "Gryffindor": "red",
        "Slytherin":  "green",
        "Ravenclaw":  "blue",
        "Hufflepuff": "gold",
    }

    numeric_data = data.select_dtypes(include=np.number)
    numeric_data = numeric_data.drop(columns=["Index"], errors="ignore")
    course = numeric_data.columns.tolist()

    if not course:
        print("No features available for pair plot.")
        return

    num_features = len(course)
    fig, axs = plt.subplots(num_features, num_features, figsize=(17, 9))

    for i in range(num_features):
        for j in range(num_features):
            ax = axs[i, j]
            if i == j:
                for house, color in house_colors.items():
                    house_data = numeric_data[data["Hogwarts House"] == house]
                    ax.hist(
                        house_data.iloc[:, i].dropna(),
                        bins=15,
                        color=color,
                        alpha=0.5,
                        label=house
                    )
            else:
                for house, color in house_colors.items():
                    house_data = numeric_data[data["Hogwarts House"] == house]
                    ax.scatter(
                        house_data.iloc[:, j],
                        house_data.iloc[:, i],
                        color=color,
                        label=house,
                        alpha=0.5,
                        s=1,
                    )

            if j == 0:
                ax.set_ylabel(numeric_data.columns[i].split(" ")[0],
                              fontsize=8)
            if i == num_features - 1:
                ax.set_xlabel(numeric_data.columns[j], fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    df = open_file(drop_na=True)
    pair_plot(df)


if __name__ == "__main__":
    main()
