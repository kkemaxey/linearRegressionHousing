# Comparing Three Implementations of Linear Regression

This project implements and compares three approaches to solving a linear regression problem:

1. **Statistical regression using covariance and variance**
2. **The normal equation using matrix operations**
3. **Gradient descent implemented from scratch**

All three approaches are applied to the same simple linear regression problem using the **RM** feature from the Boston Housing dataset to predict **MEDV**, the median home value in thousands of dollars.

The purpose of this project is not to build the most accurate housing price predictor. Instead, it is to understand how different mathematical and computational approaches solve the same regression problem and how their performance and scalability differ.

---

## Project Overview

The underlying model for each implementation is:

$$
y = mx + b
$$

where:

* $x$ = average number of rooms per dwelling (`RM`)
* $y$ = median home value (`MEDV`)
* $m$ = slope
* $b$ = y-intercept

Each method attempts to determine the values of $m$ and $b$ that best fit the data.

The project intentionally uses only one feature to keep the mathematical comparison between the three implementations clear and tractable.

---

## Methods

### 1. Statistical Approach

The first implementation calculates the slope using the relationship between covariance and variance:

$$
m = \frac{\text{Cov}(x,y)}{\text{Var}(x)}
$$

The intercept is then calculated using:

$$
b = \bar{y} - m\bar{x}
$$

This approach directly computes the parameters of the best-fit line using statistical properties of the dataset.

**Complexity:**

* Time: $O(n)$
* Space: $O(1)$

where $n$ is the number of data points.

---

### 2. Matrix Approach

The second implementation uses the normal equation:

$$
\theta = (X^TX)^{-1}X^Ty
$$

The feature matrix is augmented with a column of ones to account for the intercept.

The resulting parameter vector contains:

$$
\theta =
\begin{bmatrix}
b \
m
\end{bmatrix}
$$

This approach demonstrates how linear regression can be expressed as a system of matrix operations.

For this implementation, the model contains one feature plus the intercept term, so the feature dimension remains constant.

**Complexity as implemented:**

* Time: $O(n)$ when the feature dimension is constant
* Space: $O(n)$

For a generalized implementation with $d$ features, the matrix operations become increasingly expensive as $d$ grows.

---

### 3. Machine Learning Approach: Gradient Descent

The third implementation learns the slope and intercept iteratively using gradient descent.

The model begins with initial values for the weight and bias and repeatedly performs the following steps:

1. Generate predictions
2. Calculate the model's error
3. Calculate the gradients
4. Update the weight and bias
5. Repeat for a fixed number of epochs

The cost function is Mean Squared Error:

$$
L(w,b) =
\frac{1}{2n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

The parameters are updated using:

$$
w = w - \epsilon \frac{\partial L}{\partial w}
$$

$$
b = b - \epsilon \frac{\partial L}{\partial b}
$$

The implementation uses:

* Learning rate: `0.025`
* Epochs: `10,000`
* Training/test split: `80/20`

**Complexity:**

* Time: $O(nk)$
* Space: $O(n)$

where:

* $n$ = number of training samples
* $k$ = number of training epochs

---

## Results

The three implementations converged to nearly identical regression parameters:

| Method           | Slope | Intercept |
| ---------------- | ----: | --------: |
| Statistical      | ~9.10 |   ~-34.67 |
| Matrix           | ~9.10 |   ~-34.67 |
| Gradient Descent | ~9.11 |   ~-34.67 |

The gradient descent implementation converged to a solution within approximately **0.1%** of the analytical solutions.

This result demonstrates that gradient descent can successfully approximate the same regression solution produced directly by the statistical and matrix-based methods.

---

## Model Evaluation

The gradient descent model was evaluated using:

* **R²**
* **Root Mean Squared Error (RMSE)**

The model achieved approximately:

* **R²:** `0.419`
* **RMSE:** `6.8`

Because MEDV is measured in thousands of dollars, the RMSE corresponds to an average prediction error of approximately **$6,800**.

The R² value indicates that the model explains approximately **42% of the variance** in median home value using only the average number of rooms as a feature.

This result is expected to be limited because housing prices depend on many variables beyond the number of rooms.

---

## Complexity Comparison

| Method           | Time Complexity                       | Space Complexity |
| ---------------- | ------------------------------------- | ---------------- |
| Statistical      | $O(n)$                                | $O(1)$           |
| Matrix           | $O(n)$ for constant feature dimension | $O(n)$           |
| Gradient Descent | $O(nk)$                               | $O(n)$           |

For this specific one-feature implementation, all three approaches are approximately linear with respect to the number of samples.

However, their scalability differs as the number of features increases.

### Analytical methods

The normal equation requires operations involving:

$$
(X^TX)^{-1}
$$

As the number of features grows, matrix inversion becomes increasingly expensive.

The generalized computational complexity is dominated by the number of features rather than simply the number of data points.

### Gradient descent

Gradient descent avoids explicitly inverting a potentially large feature matrix.

Its computational cost scales with both:

* the number of training samples
* the number of optimization iterations

This makes iterative optimization more attractive for problems with many features, although it introduces additional considerations such as:

* learning rate selection
* convergence
* number of iterations
* initialization

---

## Key Takeaways

This project demonstrated several important differences between analytical and iterative approaches to linear regression.

### Closed-form methods are efficient for low-dimensional problems

For a problem with only one feature, the statistical and matrix methods can calculate the solution directly and efficiently.

Running 10,000 gradient descent epochs to reach a solution that can be computed analytically in a single calculation is unnecessary for this specific problem.

### Gradient descent is more flexible for larger problems

The advantage of gradient descent becomes more apparent as the number of features increases.

Rather than directly calculating a matrix inverse, gradient descent iteratively optimizes the model parameters.

This can make it more practical for higher-dimensional problems, although the tradeoff is additional iteration and hyperparameter tuning.

### Different implementations can solve the same mathematical problem

The three methods appear different at the implementation level, but they are ultimately optimizing the same underlying least-squares regression objective.

The main difference is how they reach the solution:

* The statistical method derives the parameters directly from covariance and variance.
* The matrix method expresses the solution using linear algebra.
* Gradient descent approximates the solution through iterative optimization.

---

## Limitations

This project has several important limitations.

### Single-feature model

The model only uses `RM`, the average number of rooms per dwelling.

Housing prices are influenced by many other factors, including:

* location
* crime rates
* accessibility
* property characteristics
* tax rates
* socioeconomic factors

The relatively low R² score reflects the limitations of using a single feature.

### Different training procedures

The analytical methods use the full dataset, while the gradient descent implementation uses an 80/20 train/test split.

As a result, the experimental comparison is not perfectly identical between methods.

A stronger experimental design would compare:

1. All three methods using the same training data
2. All three methods using the same test data for evaluation

### Dataset limitations

The Boston Housing dataset contains only 506 samples and originates from the 1970 U.S. Census.

The dataset has also been deprecated from `scikit-learn` because of ethical concerns surrounding one of its features.

It is used in this project strictly as a benchmark dataset for comparing regression implementations and computational methods—not as a basis for modern housing-market conclusions.

### Gradient descent hyperparameters

The learning rate and number of epochs were selected experimentally.

The implementation does not currently use an automatic convergence criterion or systematic hyperparameter search.

---

## Repository Structure

```text
.
├── ML_Model/
│   └── Saved model parameters and machine learning outputs
├── Scatter_Plots/
│   └── Visualizations comparing regression results
├── Linear_Regression_Poster.png
├── Methods of Linear Regression Documentation.docx
└── linear_regression_housing.py
```

---

## Running the Project

### Requirements

Python 3.8+

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run

```bash
python linear_regression_housing.py
```

The script will:

1. Load the Boston Housing dataset
2. Extract the `RM` and `MEDV` features
3. Calculate the regression parameters using covariance and variance
4. Calculate the regression parameters using the normal equation
5. Train a regression model using gradient descent
6. Compare the resulting parameters
7. Generate visualizations
8. Save model-related outputs

---

## Project Documentation

For a detailed explanation of the mathematical derivations, implementation decisions, complexity analysis, results, and limitations, see:

**[Methods of Linear Regression Documentation](./Methods%20of%20Linear%20Regression%20Documentation.docx)**

The project poster is also included in the repository:

**[Linear Regression Poster](./Linear_Regression_Poster.png)**

---

## Conclusion

This project compares three different implementations of linear regression to demonstrate that the same mathematical objective can be approached through statistical formulas, matrix algebra, or iterative optimization.

For the low-dimensional problem used here, the analytical methods are the most direct and computationally efficient. However, gradient descent provides a more flexible optimization framework as the dimensionality of a problem increases.

The primary goal of this project was therefore not to produce the most accurate housing price model, but to understand the mathematical foundations, implementation tradeoffs, and scalability considerations behind different approaches to linear regression.
