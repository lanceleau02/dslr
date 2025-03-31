import sys

def describe(dataset):
	

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 describe.py <dataset>")
		sys.exit(1)
	describe(sys.argv[1])

if __name__ == "__main__":
	main()