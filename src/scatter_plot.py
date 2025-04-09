from src import pd, plt, combinations

def all_scatter_plot(df):
	# Select only numeric columns, excluding the 'Index' column
	numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Index'])
	# Get the list of numeric features/column names
	features = numeric_df.columns
	# Calculate total number of features
	num_features = len(features)
	# Calculate total number of unique feature pairs for scatter plots
	num_plots = num_features * (num_features - 1) // 2
	# Define layout: number of columns in the subplot grid
	num_cols = 10
	# Calculate required number of rows based on number of plots
	num_rows = (num_plots + num_cols - 1) // num_cols
	# Create subplots grid with appropriate size
	fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
	# Flatten axes array for easy iteration
	axes = axes.flatten()
	# Get unique Hogwarts houses and define corresponding colors
	houses = df['Hogwarts House'].unique()
	colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}
	# Iterate over all unique pairs of numeric features
	for ax, (feature1, feature2) in zip(axes, combinations(features, 2)):
		# Plot scatter points for each house
		for house in houses:
			# Filter data for the current house
			house_data = df[df['Hogwarts House'] == house]
			# Plot the feature pair for this house
			ax.scatter(house_data[feature1], house_data[feature2], alpha=0.5, s=3, label=house, color=colors[house])
		# Set plot title and axis labels with small font size
		ax.set_title(f'{feature1} vs {feature2}', fontsize=5)
		ax.set_xlabel(feature1, fontsize=5)
		ax.set_ylabel(feature2, fontsize=5)
		# Add a grid for readability
		ax.grid(True)
	# Remove any unused subplot axes (in case there are more axes than plots)
	for ax in axes[num_plots:]:
		fig.delaxes(ax)
	# Adjust spacing between subplots
	plt.tight_layout()
	plt.subplots_adjust(hspace=1, bottom=0.05)
	# Display the plots
	plt.show()
	
def scatter_plot(df):
	# Reuse top correlated pair
	feature1, feature2 = 'Defense Against the Dark Arts', 'Astronomy'
	# Plot each house with a different color
	houses = df['Hogwarts House'].unique()
	colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}
	plt.figure(figsize=(8,6))
	for house, color in zip(houses, colors):
		subset = df[df['Hogwarts House'] == house]
		plt.scatter(subset[feature1], subset[feature2], label=house, alpha=0.6, color=colors[house])
	plt.xlabel(feature1)
	plt.ylabel(feature2)
	plt.title(f'{feature1} vs {feature2} by Hogwarts House')
	plt.legend()
	plt.grid(True)
	plt.show()

def calculate_correlation(df):
	# Calculate the absolute correlation matrix between all numeric columns
	correlation_matrix = df.corr(numeric_only=True).abs()
	# Save the correlation matrix as a text file
	with open("./data/correlation_matrix.txt", "w") as f:
		f.write(correlation_matrix.to_string())
	# "Unstack" the matrix to turn it into a Series of pairs: (feature1, feature2) -> correlation
	corr_pairs = correlation_matrix.unstack()
	# Sort correlation pairs in descending order using Quicksort
	sorted_pairs = corr_pairs.sort_values(kind='quicksort', ascending=False)
	# Remove self-correlations (where feature1 == feature2), which always equal 1.0
	filtered_pairs = sorted_pairs[sorted_pairs < 1.0]
	# Get the pair of features with the highest correlation
	most_similar = filtered_pairs.idxmax()
	highest_corr_value = filtered_pairs.max()
	# Print the result
	print(f"Most similar pair: {most_similar} with correlation {highest_corr_value:.2f}")

def main():
	df = pd.read_csv("./data/dataset_train.csv")
	all_scatter_plot(df)
	scatter_plot(df)
	calculate_correlation(df)

if __name__ == "__main__":
	main()