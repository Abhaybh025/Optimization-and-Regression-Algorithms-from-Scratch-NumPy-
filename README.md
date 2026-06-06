# Linear Regression from Scratch — Math to Optimization
A complete from-scratch implementation of Multiple Linear Regression and Gradient Descent in NumPy, starting from the closed-form Normal Equation and progressively building up to Batch, Stochastic, and Mini-Batch Gradient Descent. No sklearn for modeling — only for dataset loading and validation.

> **Goal:** Bridge the gap between theory and practice — understand the math behind linear regression, then understand *how* and *why* gradient descent variants behave differently as an optimization strategy.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Part 1 — Normal Equation (Closed-Form Solution)](#part-1--normal-equation-closed-form-solution)
4. [Part 2 — Gradient Descent Variants](#part-2--gradient-descent-variants)
5. [Key Observations](#key-observations)
6. [Visualizations](#visualizations)
7. [Usage](#usage)

---

## Project Structure

```
├── MultiLinearRegression.ipynb       # Part 1: Normal Equation implementation
├── GradientDescentVariants.ipynb     # Part 2: BGD, SGD, MBGD from scratch
├── requirements.txt
└── README.md
```

---

## Dataset

| Dataset | Source | Samples | Features | Target |
|---------|--------|---------|----------|--------|
| California Housing | `sklearn.datasets.fetch_california_housing` | 20,640 | 8 (median income, housing age, avg rooms, etc.) | Median house value |

> All features and target are **standardized** for stable gradient descent convergence across both parts.

---

## Part 1 — Normal Equation (Closed-Form Solution)

### What it does
Implements a `MultiLinearReg` class from scratch using the **Normal Equation**:

$$\hat{\theta} = (X^T X)^{-1} X^T y$$

No iterative updates, no learning rate — the optimal coefficients are computed directly via matrix operations.

### Class API

```python
model = MultiLinearReg()
model.fit(X_train, y_train)       # Computes coefficients via Normal Equation
predictions = model.predict(X)    # Matrix multiplication with learned weights
r2  = model.r2_score(y, y_pred)
mse = model.mse(y, y_pred)
mae = model.mae(y, y_pred)
residuals = model.residuals(y, y_pred)
```

### Results — Scratch vs sklearn Validation

| Metric | Scratch Implementation | sklearn LinearRegression |
|--------|----------------------|--------------------------|
| R² Score | ✅ Matches | Reference |
| MSE | ✅ Matches | Reference |
| MAE | ✅ Matches | Reference |

> Coefficients match sklearn's output, confirming the matrix math is correct.

### Notebook Structure
1. Introduction & Dataset Overview
2. Data Exploration — preview, stats, feature distributions
3. Model Implementation — `MultiLinearReg` class
4. Training & Evaluation — comparison with sklearn on California Housing
5. Residual Analysis & Visualization
6. Conclusion & Insights

---

## Part 2 — Gradient Descent Variants

### What it does
Implements four optimization strategies from scratch on the same regression task, then compares their convergence behavior and final coefficients.

| Model | Update Rule | Data Used Per Step |
|-------|-------------|-------------------|
| **OLS** | Normal Equation (closed-form) | Full dataset, one shot |
| **BGD** | Gradient on full dataset | All N samples |
| **MBGD** | Gradient on random mini-batch | `batch_size` samples |
| **SGD** | Gradient on single sample | 1 sample |

### Convergence Behavior

```
BGD   → smooth, steady convergence — predictable but slow on large data
MBGD  → near-smooth, fastest practical convergence — best of both worlds  
SGD   → noisy fluctuations per epoch, but final loss matches others
OLS   → instant, but not scalable (matrix inversion is O(n³))
```

### Key Hyperparameters Explored
- **Learning rate** — too high → diverges, too low → slow convergence
- **Batch size** (MBGD) — controls the speed/stability tradeoff
- **Epochs** — when to stop; all variants converge to similar final loss

---

## Key Observations

**1. Coefficient Convergence**
All gradient descent variants converge to coefficients very close to the OLS closed-form solution, confirming correctness of the gradient math.

**2. Intercept Behavior**
SGD shows slight intercept fluctuation due to the stochasticity of single-sample updates. This is expected and doesn't meaningfully affect predictions.

**3. Loss Curves**
- BGD → perfectly smooth MSE curve
- MBGD → nearly smooth, very close to BGD
- SGD → visibly noisy but converges to the same final loss

**4. Scalability**
The Normal Equation becomes expensive at scale due to matrix inversion. Gradient Descent variants scale linearly with data size, making them the practical choice for large datasets.

---

## Visualizations

| Plot | What it shows |
|------|--------------|
| Predicted vs Actual | Scatter plot comparing all model outputs against ground truth |
| Coefficients Comparison | Bar chart — all 4 models side by side |
| Intercept Comparison | Bar chart highlighting SGD stochasticity |
| Loss Convergence | MSE vs Epochs for BGD, MBGD, and SGD |
| Residual Distribution | Histogram of residuals from the Normal Equation model |
| Target Distribution | Distribution of median house values |

---

## Usage

```bash
# Clone the repository
git clone https://github.com/Abhaybh025/YOUR_REPO_NAME
cd YOUR_REPO_NAME

# Open either notebook
jupyter notebook MultiLinearRegression.ipynb
jupyter notebook GradientDescentVariants.ipynb
```

Or open directly in **Google Colab** — no local setup needed.

---

## License
Open source — free to use, adapt, or extend for educational purposes.
