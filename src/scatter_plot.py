from src import pd, plt, np

def scatter_plot(datafile):
	df = pd.read_csv("./data/dataset_train.csv")
	
	# Define house colors
	house_colors = {
		"Gryffindor": "red",
		"Hufflepuff": "yellow",
		"Ravenclaw": "blue",
		"Slytherin": "green"
	}

	""" correlation = df.groupby("")[["Herbology", "Defense Against the Dark Arts"]].corr()
	print(f"Pearson Correlation: {correlation:.2f}") """

	# Compute the correlation matrix (absolute values to consider both positive and negative correlations)
	correlation_matrix = df.corr(numeric_only=True)

	# Find the indices of the most similar pair of features
	# 1. correlation_matrix.values converts the Pandas DataFrame into a NumPy array (2D matrix)
	# 2. np.argmax() finds the index of the maximum value in the flattened (1D) version of the correlation matrix.
	# 3. np.unravel_index() converts that 1D index back into 2D indices (row, column) of the original matrix.
	np.fill_diagonal(correlation_matrix.values, 0) # Ignore the diagonal values (self-correlation) by setting them to 0
	most_similar = np.unravel_index(np.argmax(correlation_matrix.values), correlation_matrix.shape)

	# Retrieve the corresponding feature names
	feature_1, feature_2 = df.columns[most_similar[0]], df.columns[most_similar[1]]

	# Plot the scatter plot of the two most similar features
	plt.figure(figsize=(8, 6))
	for house, color in house_colors.items():
		subset = df[df["Hogwarts House"] == house]  # Filter by house
		plt.scatter(subset[feature_1], subset[feature_2], color=color, label=house, alpha=0.6)
	plt.xlabel(feature_1)
	plt.ylabel(feature_2)
	plt.title(f"Scatter Plot: {feature_1} vs {feature_2}")
	plt.grid(True)
	plt.show()

def main():
	scatter_plot("./data/dataset_train.csv")

if __name__ == "__main__":
	main()