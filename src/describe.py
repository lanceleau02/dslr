from src import sys, pd, np

def describe(dataset):
	df = pd.read_csv(dataset)
	df = df.loc[:, df.columns != 'Index']
	stats = {}
	for col in df.columns:
		if df[col].dtype in [np.float64, np.int64]:
			data = df[col].dropna().values
			n = sum(1 for _ in data)
			sorted_data = sorted(data)
			if (n == 0):
				continue
			mean = sum(sorted_data) / n
			variance = sum((x - mean) ** 2 for x in sorted_data) / (n - 1)
			std_dev = variance ** 0.5
			min_val = sorted_data[0]
			max_val = sorted_data[-1]
			q25 = sorted_data[int(0.25 * (n - 1))]
			q50 = sorted_data[int(0.50 * (n - 1))]
			q75 = sorted_data[int(0.75 * (n - 1))]
			
			stats[col] = {
				"Count": n, # The total number of valid (non-NaN) values in the column.
				"Mean": mean, # The average (arithmetic mean) of all values in the column.
				"Std": std_dev, # The standard deviation, which measures how much the values deviate from the mean. A higher value indicates more spread-out data.
				"Min": min_val, # The smallest value in the column.
				"25%": q25, # The first quartile (Q1), which represents the value below which 25% of the data falls.
				"50%": q50, # The median (Q2), the middle value that separates the dataset into two equal halves.
				"75%": q75, # The third quartile (Q3), meaning 75% of the data falls below this value.
				"Max": max_val # The largest value in the column.
			}
	print(pd.DataFrame(stats))

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 describe.py <dataset>")
		sys.exit(1)
	describe(sys.argv[1])

if __name__ == "__main__":
	main()