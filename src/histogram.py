from src import sys

def histogram():
	

def main():
	if len(sys.argv) != 2:
		print("Usage: python3 histogram.py <dataset>")
		sys.exit(1)
	histogram(sys.argv[1])

if __name__ == "__main__":
	main()