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

## Understand the goal

To truly understand what we need to build, we are going to look at it not as a Python script, but as a magical factory. The goal was to build a Mechanical Sorting Hat. Because computers cannot read minds, the mechanical hat has to rely entirely on math and historical data (the student's grades).

### Phase 1: The Admissions Office (Data Preprocessing)
Before the Mechanical Sorting Hat can even look at a student, the raw paperwork has to be perfectly organized. 

* **The Problem:** The raw CSV file is a mess. Some students missed exams (NaNs), and the grading scales are completely wild. Astronomy is graded on a scale of -1000 to 1000, while Potions is graded from 0 to 10. If we feed this directly to the machine, it will assume Astronomy is infinitely more important just because the numbers are bigger.
* **The Solution (Imputation & Scaling):**
  * First, we fill in the blank test scores with the class average (`fillna`). 
  * Then, we send the grades through **The Great Equalizer** (Z-Score Standardization). We subtract the mean ($\mu$) and divide by the standard deviation ($\sigma$). 
* **The Metaphor:** Imagine taking every student's test score and translating it into a universal "Z-Language". A `0` means you are perfectly average. A `1.5` means you are above average. Now, an outstanding Potions grade and an outstanding Astronomy grade look exactly the same to the machine. 
* **The Golden Rule:** We lock the $\mu$ and $\sigma$ blueprints in a safe (`scaling_params.json`). When new students arrive next year, we MUST judge them using these exact same standards, or the whole system breaks.

### Phase 2: The Mechanical Brain (Core Math Helpers)
Next, we built the three mechanical gears that allow the machine to actually "think" and learn from its mistakes.

**1. The Probability Lens (`sigmoid`)**
Linear regression outputs raw, wild numbers (like $z = 450$). But we need a percentage chance (from $0$ to $1$). The Sigmoid function is a magical lens. No matter how massive or negative a number you push through it, it squashes it into a neat probability. (e.g., $0.85$ means an 85% chance of being in Gryffindor).

**2. The Guilt-O-Meter (`compute_cost` / Log Loss)**
The machine needs to know when it messes up. If it looks at Harry Potter, predicts he is 99% likely to be a Slytherin, and then checks the answer key and sees he is actually a Gryffindor, the Guilt-O-Meter penalizes it massively. It calculates the mathematical distance between the machine's *guesses* and the *actual truth*.

**3. The Blindfolded Hiker (`gradient_descent`)**
This is how the machine learns. 
* **The Metaphor:** Imagine the machine is blindfolded, standing on the side of a massive crater. The bottom of the crater represents perfect accuracy (zero error). 
* It takes a guess (calculates probabilities).
* It checks the Guilt-O-Meter to see its altitude.
* It feels the ground with its foot to find the slope (calculating the **Gradient** using matrix multiplication).
* It takes a step downhill (the size of the step is your **Learning Rate**, $\alpha$).
* It repeats this 1,000 times until it reaches the flat bottom of the crater. The coordinates of the bottom are your perfect $\theta$ weights!

### Phase 3: The Four Bouncers (One-vs-All Training)
Here is the biggest secret of Logistic Regression: **It can only answer Yes or No.** It cannot pick between four houses. So, how did we solve this?

* **The Metaphor:** Instead of building one omniscient Sorting Hat, we hired **Four Specialized Bouncers**.
    * We train Bouncer 1 to only care about one thing: *"Are you a Gryffindor, YES or NO?"* (All other houses are grouped into "No").
    * We train Bouncer 2: *"Are you a Slytherin, YES or NO?"*
    * Bouncer 3 looks for Ravenclaws.
    * Bouncer 4 looks for Hufflepuffs.
* **The Code:** We loop through the four houses. For each house, we set up our blindfolded hiker, run gradient descent 1,000 times, and find the perfect weights for that specific bouncer. We then write down all four sets of weights in our ledger (`weights_data.json`).

### Phase 4: The Final Exam (Prediction)
The training is over. It is time to test the machine on brand new students (`dataset_test.csv`) who do not have a house assigned to them yet.

* **Step 1 - The Equalizer Returns:** The new students walk in. We immediately open our safe, pull out last year's `scaling_params.json`, and translate their grades into the universal Z-Language so the machine can understand them.
* **Step 2 - The Interview Panel:** The student stands in front of the Four Bouncers. 
    * The Gryffindor bouncer looks at the grades, applies his weights, and says: *"I am 20% sure this is a Gryffindor."*
    * The Slytherin bouncer says: *"I am 85% sure this is a Slytherin."*
    * The Ravenclaw bouncer says: *"I am 10% sure."*
    * The Hufflepuff bouncer says: *"I am 2% sure."*
* **Step 3 - The Verdict:** Using Pandas' `idxmax()`, the machine simply points to the bouncer who shouted the highest number. The student is branded a Slytherin, written into `houses.csv`, and sent to the dungeons.

## License

This project is licensed under the **42 School** License.

- **Educational Use Only**: This project is intended for educational purposes at the 42 School as part of the curriculum.
- **Non-commercial Use**: The code may not be used for commercial purposes or redistributed outside of the 42 School context.
- **No Warranty**: The project is provided "as-is", without any warranty of any kind.

For more details, see the [LICENSE](https://github.com/lanceleau02/dslr/blob/main/LICENSE) file.