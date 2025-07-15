from src import pd, sns, plt

def pair_plot(df):
	# Ensure 'Hogwarts House' is treated as a category
	df['Hogwarts House'] = df['Hogwarts House'].astype('category')

	# Only use numeric features + House
	numeric_df = df.select_dtypes(include='number')
	numeric_df['Hogwarts House'] = df['Hogwarts House']

	# Define custom palette
	house_colors = {
		"Gryffindor": "red",
		"Ravenclaw": "blue",
		"Hufflepuff": "gold",
		"Slytherin": "green"
	}

	# Pair plot with hue based on house
	g = sns.pairplot(numeric_df, hue='Hogwarts House', palette=house_colors, diag_kind='hist')
	# Adjust subplot layout via the Figure object
	g.fig.subplots_adjust(left=0.04, right=0.94, top=0.98, bottom=0.04, wspace=0.05, hspace=0.05)
	plt.show()

def main():
	df = pd.read_csv("./data/dataset_train.csv")
	pair_plot(df)

if __name__ == "__main__":
	main()
