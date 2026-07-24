# Comparing Three Implementations of Linear Regression

A from-scratch comparison of three approaches to solving the same linear regression problem:

* **Statistical estimation** using covariance and variance
* **Closed-form linear algebra** using the normal equation
* **Iterative optimization** using gradient descent

The goal was to understand how these approaches differ in implementation, computational complexity, scalability, and numerical behavior.

---

## TL;DR

All three implementations converge to essentially the same linear regression solution:

| Method                | Slope | Intercept |
| --------------------- | ----: | --------: |
| Covariance / Variance | ~9.10 |   ~-34.67 |
| Normal Equation       | ~9.10 |   ~-34.67 |
| Gradient Descent      | ~9.11 |   ~-34.67 |

The gradient descent implementation converged to within approximately **0.1%** of the analytical solutions.

### Main conclusion

For this low-dimensional problem, the closed-form methods are the most direct solution.

Gradient descent requires significantly more computation for this specific dataset, but its iterative optimization approach becomes more attractive as the number of features and problem dimensionality increase.

---

## Why I Built This

Linear regression is often introduced through a library call:

```python
model.fit(X, y)
```

This project was an attempt to understand what happens underneath that abstraction.

Rather than using a prebuilt regression implementation, I implemented three approaches to the same problem and compared how each one:

* calculates model parameters
* scales with input size
* uses memory
* converges to a solution
* behaves as the dimensionality of the problem changes

The goal was to compare **algorithms and implementation strategies**, not to build a production housing-price predictor.

---

## Problem

The model uses a single feature from the Boston Housing dataset:

* **Input (`RM`)**: average number of rooms per dwelling
* **Target (`MEDV`)**: median home value, measured in thousands of dollars

The model is:

$$
\hat{y} = wx + b
$$

The one-feature restriction was intentional. It makes it possible to compare the mathematical approaches directly while keeping the underlying regression problem easy to inspect.

---

## Implementations

### 1. Covariance / Variance

The slope is calculated directly:

$$
m = \frac{\text{Cov}(x,y)}{\text{Var}(x)}
$$

The intercept is then:

$$
b = \bar{y} - w\bar{x}
$$

This produces the least-squares solution directly for the single-feature problem.

**Complexity:**

* Time: `O(n)`
* Additional space: `O(1)`

---

### 2. Normal Equation

The second implementation uses the closed-form matrix solution:

$$
\theta = (X^TX)^{-1}X^Ty
$$

The input matrix is augmented with a column of ones to represent the intercept.

This approach demonstrates how linear regression can be represented as a linear algebra problem.

For the current implementation, the number of features is constant, so the dominant cost scales linearly with the number of samples.

**Complexity for this implementation:**

* Time: `O(n)`
* Space: `O(n)`

**Scaling consideration:**

As the number of features increases, matrix operations become increasingly expensive. Explicit matrix inversion also introduces numerical-stability concerns, making direct linear solves or decompositions preferable in production numerical code.

---

### 3. Gradient Descent

The final implementation learns the parameters iteratively.

Each epoch:

1. Generates predictions
2. Calculates the loss
3. Computes the gradients
4. Updates the weight and bias

The objective is Mean Squared Error:

$$
L(w,b) =
\frac{1}{2n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

The parameters are updated using:

$$
w \leftarrow w - \alpha \frac{\partial L}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial L}{\partial b}
$$

Configuration:

* Learning rate: `0.025`
* Epochs: `10,000`
* Training split: `80%`
* Test split: `20%`

**Complexity:**

* Time: `O(nk)`
* Space: `O(n)`

where:

* `n` = number of training samples
* `k` = number of epochs

---

## Results

The analytical approaches produced effectively identical solutions.

Gradient descent converged to a nearly identical solution after iterative optimization:

```text
Analytical slope:       ~9.10
Gradient descent slope: ~9.11
```

The difference was approximately **0.1%**.

This validates that the gradient descent implementation is optimizing the same underlying least-squares objective as the closed-form solutions.

---

## Model Performance

Using the gradient descent implementation:

| Metric | Result |
| ------ | -----: |
| R²     | ~0.419 |
| RMSE   |   ~6.8 |

The model explains approximately **42% of the variance** in median home value using only the average number of rooms as an input.

The RMSE corresponds to approximately **$6,800** in the dataset's units.

The relatively limited predictive performance is expected. Housing prices depend on substantially more than the number of rooms in a property.

---

## Complexity Comparison

| Method                | Time    | Space  | Solution  |
| --------------------- | ------- | ------ | --------- |
| Covariance / Variance | `O(n)`  | `O(1)` | Direct    |
| Normal Equation       | `O(n)`* | `O(n)` | Direct    |
| Gradient Descent      | `O(nk)` | `O(n)` | Iterative |

*For the current implementation with a constant number of features.

The most important result is that the complexity comparison changes as the problem dimension changes.

For this project:

```text
number of features = 1
```

Therefore, the matrix-based methods appear approximately linear with respect to the number of samples.

For higher-dimensional problems, the tradeoff becomes more significant:

* Closed-form methods perform increasingly expensive matrix operations.
* Gradient descent trades exact computation for iterative optimization.
* Gradient descent introduces additional concerns around convergence and hyperparameter selection.

---

## Engineering Tradeoffs

### Closed-form methods

**Advantages**

* Direct solution
* No learning-rate tuning
* No convergence loop
* Efficient for small feature spaces

**Disadvantages**

* Matrix operations become expensive as dimensionality increases
* Explicit inversion can introduce numerical instability
* Memory requirements can grow significantly

### Gradient descent

**Advantages**

* Avoids explicitly computing a matrix inverse
* Scales naturally with the number of training examples and optimization steps
* Extensible to larger and more complex optimization problems

**Disadvantages**

* Requires hyperparameter selection
* Requires an iterative convergence process
* Can converge slowly or fail to converge with poor configuration

---

## What I Would Improve

This experiment exposed several weaknesses in the initial implementation.

### 1. Standardize the experimental setup

The analytical methods use the full dataset, while gradient descent uses an 80/20 split.

A stronger comparison would evaluate all three methods using the same:

* training set
* test set
* evaluation metrics

This would separate implementation differences from differences caused by the data each method receives.

### 2. Replace explicit matrix inversion

The current normal-equation implementation directly computes:

```text
(XᵀX)⁻¹
```

A production-oriented implementation should prefer solving the linear system directly or using a numerically stable decomposition such as QR or SVD.

### 3. Add convergence-based stopping

The gradient descent implementation currently trains for a fixed number of epochs.

A stronger implementation would stop when the change in loss falls below a defined tolerance.

### 4. Improve test coverage

The implementations should be tested against:

* known synthetic datasets
* constant inputs
* small datasets
* degenerate inputs
* comparisons against a trusted reference implementation

### 5. Generalize the model interface

The next version would separate:

* data loading
* model implementations
* evaluation
* visualization
* experiment configuration

This would make it easier to extend the project to multiple features and additional optimization methods.

---

## Limitations

This project is primarily an implementation and algorithm-comparison exercise.

The results should not be interpreted as a modern housing-market analysis.

Important limitations include:

* The model uses only one feature.
* The dataset contains only 506 samples.
* The Boston Housing dataset is historically derived and has documented ethical concerns.
* The gradient descent hyperparameters were selected experimentally.
* The three methods were not initially evaluated using identical train/test procedures.
* The normal-equation implementation uses explicit matrix inversion.

These limitations are important because the goal of the project is to understand the computational approaches, not to make real-world predictions about housing prices.

---

## Repository Structure

```text
.
├── ML_Model/
│   └── Saved model outputs
├── Scatter_Plots/
│   └── Regression visualizations
├── Linear_Regression_Poster.png
├── Methods of Linear Regression Documentation.docx
└── linear_regression_housing.py
```

---

## Running the Project

### Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run

```bash
python linear_regression_housing.py
```

The program:

1. Loads the dataset
2. Extracts the input and target variables
3. Calculates the analytical regression solution
4. Calculates the matrix-based solution
5. Trains the gradient descent model
6. Compares the resulting parameters
7. Generates visualizations
8. Saves model outputs

---

## Documentation

For the full mathematical derivations, complexity analysis, results, and limitations:

📄 [Read the full technical documentation](./Methods%20of%20Linear%20Regression%20Documentation.docx)

🖼️ [View the project poster](./Linear_Regression_Poster.png)

---

## Key Takeaway

The most important lesson from this project was not that one regression implementation is universally better than the others.

The lesson was that **the best algorithm depends on the structure of the problem**.

For a low-dimensional regression problem, a closed-form solution can be simple and efficient.

As the dimensionality and complexity of the problem increase, iterative optimization methods such as gradient descent can offer a more scalable alternative.

This project was an exercise in understanding that tradeoff by implementing the underlying algorithms rather than treating linear regression as a black box.
