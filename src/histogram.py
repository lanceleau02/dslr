from src import pd, plt, np

def histogram(dataset):
	# Load dataset
	df = pd.read_csv(dataset)  # Replace with your actual file

	# Identify numerical columns (excluding "Hogwarts House" and non-numeric columns)
	courses = df.select_dtypes(include=["number"]).columns
	courses = [col for col in courses if col.lower() != "index"]  # Exclude "Index" if present

	# Define house colors
	house_colors = {
		"Gryffindor": "red",
		"Hufflepuff": "yellow",
		"Ravenclaw": "blue",
		"Slytherin": "green"
	}

	# Determine grid size (rows x cols)
	num_courses = len(courses)
	cols = min(3, num_courses)  # Maximum of 3 columns
	rows = int(np.ceil(num_courses / cols))  # Adjust rows dynamically

	# Create a large figure
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))  # Adjust figure size

	# Flatten axes array for easy iteration (handles single-row cases)
	axes = np.array(axes).flatten()

	# Plot each course in its respective subplot
	for i, course in enumerate(courses):
		ax = axes[i]
		for house, color in house_colors.items():
			house_data = df[df["Hogwarts House"] == house][course].dropna()
			ax.hist(house_data, bins=20, alpha=1.0, color=color, density=True)

		# Customize each subplot
		ax.set_title(course)
		ax.set_xlabel("Score")
		ax.set_ylabel("Density")
		ax.grid(True)

	# Remove empty subplots (if any)
	for i in range(num_courses, len(axes)):
		fig.delaxes(axes[i])

	# Adjust layout to prevent overlapping
	plt.tight_layout()

	# Adjust layout with more space between rows
	plt.subplots_adjust(hspace=0.8, top=0.96, bottom=0.05)  # Increase space between rows

	# Create a single legend at the bottom-right
	handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=1.0) for color in house_colors.values()]
	labels = list(house_colors.keys())
	fig.legend(handles, labels, loc="lower right", title="Hogwarts Houses", frameon=True)

	# Show the plots
	plt.show()

def main():
	histogram("./data/dataset_train.csv")

if __name__ == "__main__":
	main()