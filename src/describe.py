from src import sys, pd, np
from src.utils import sum_, sort_, mean_, std_, min_, max_, percentile_

def describe(dataset):
	df = pd.read_csv(dataset)
	df = df.loc[:, df.columns != 'Index']
	stats = {}
	for col in df.columns:
		if df[col].dtype in [np.float64, np.int64]:
			data = df[col].dropna().values
			n = sum_(1 for _ in data)
			sorted_data = sort_(data)
			if (n == 0):
				continue
			mean = mean_(data)
			std = std_(data)
			min = min_(data)
			max = max_(data)
			q25 = percentile_(sorted_data, 0.25)
			q50 = percentile_(sorted_data, 0.50)
			q75 = percentile_(sorted_data, 0.75)
			
			stats[col] = {
				"Count": n,
				"Mean": mean,
				"Std": std,
				"Min": min,
				"25%": q25,
				"50%": q50,
				"75%": q75,
				"Max": max
			}
	print(pd.DataFrame(stats))

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 describe.py <dataset>")
		sys.exit(1)
	describe(sys.argv[1])

if __name__ == "__main__":
	main()