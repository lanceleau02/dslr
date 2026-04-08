<div align="center">

# dslr

**Write a classifier and save Hogwarts!**

</div>

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lanceleau02/dslr.git
```

2. Navigate to the project directory:

```bash
cd dslr
```

3. Create and install the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

## Subject Breakdown

This project introduces machine learning by implementing a **logistic regression** model from scratch. The goal is to predict **which Hogwarts house a student belongs to based on their magical subject scores** using a dataset of past students. It consists of several key scripts: `describe.py` to calculate statistical information about the dataset, visualization scripts (`histogram.py`, `scatter_plot.py`, `pair_plot.py`) to explore the data and select the best features, `logreg_train.py` to train the model, and finally, `logreg_predict.py` to sort new students into their respective houses based on the trained weights.

At its core, the model finds patterns in multidimensional data: certain combinations of grades (like high scores in Potions or Astronomy) strongly correlate with specific houses. Instead of manually writing conditional rules to sort the students, the model learns the mathematical boundaries between the four houses, then uses those learned relationships to classify new students.

The process starts with analyzing and visualizing the dataset to identify which magical subjects are actually useful for separating the students. Then, using a "One-vs-All" strategy and **gradient descent**, the model adjusts its parameters to minimize classification errors across four separate binary classifiers (e.g., Gryffindor vs. Not Gryffindor). Once trained, the model calculates the probability of a new student belonging to each house and assigns them to the highest match. This project covers the essential steps of data exploration, feature scaling, and multi-class classification, introducing fundamental data science concepts used in more complex machine learning pipelines.

## Approach & Implementation

## License

This project is licensed under the **42 School** License.

- **Educational Use Only**: This project is intended for educational purposes at the 42 School as part of the curriculum.
- **Non-commercial Use**: The code may not be used for commercial purposes or redistributed outside of the 42 School context.
- **No Warranty**: The project is provided "as-is", without any warranty of any kind.

For more details, see the [LICENSE](https://github.com/lanceleau02/dslr/blob/main/LICENSE) file.