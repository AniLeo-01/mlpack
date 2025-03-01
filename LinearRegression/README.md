# Linear Regression Implementation

## Table of Contents
- [Introduction](#introduction)
- [Theory](#theory)
  - [Linear Regression Model](#linear-regression-model)
  - [Cost Function](#cost-function)
  - [Optimization Methods](#optimization-methods)
    - [Gradient Descent](#gradient-descent)
    - [Normal Equation](#normal-equation)
- [Implementation](#implementation)
  - [Class Structure](#class-structure)
  - [Methods](#methods)
  - [Usage Example](#usage-example)
- [Performance Considerations](#performance-considerations)
- [Extensions](#extensions)

## Introduction

Linear regression is one of the most fundamental and widely used algorithms in machine learning and statistics. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.

This implementation provides a flexible and efficient linear regression model with support for different optimization methods.

## Theory

### Linear Regression Model

The linear regression model assumes that the relationship between the dependent variable y and the independent variables X is linear. For a single feature, the model can be represented as:

```
y = θ₀ + θ₁x + ε
```

Where:
- y is the dependent variable (target)
- x is the independent variable (feature)
- θ₀ is the y-intercept (bias term)
- θ₁ is the slope (weight)
- ε is the error term

For multiple features, the model extends to:

```
y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ + ε
```

In vector notation, this can be written as:

```
y = Xθ + ε
```

Where:
- y is an m×1 vector of target values
- X is an m×(n+1) matrix of feature values (with a column of ones for the bias term)
- θ is an (n+1)×1 vector of weights
- ε is an m×1 vector of error terms

### Cost Function

To find the optimal values for the parameters θ, we need to minimize a cost function. For linear regression, the most common cost function is the Mean Squared Error (MSE):

```
J(θ) = (1/2m) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Where:
- m is the number of training examples
- h_θ(x⁽ⁱ⁾) is the predicted value for the i-th example
- y⁽ⁱ⁾ is the actual value for the i-th example

The factor of 1/2 is used to simplify the derivative calculation.

### Optimization Methods

#### Gradient Descent

Gradient descent is an iterative optimization algorithm that minimizes the cost function by updating the parameters in the direction of the negative gradient:

```
θⱼ := θⱼ - α * (∂J(θ)/∂θⱼ)
```

Where:
- α is the learning rate
- ∂J(θ)/∂θⱼ is the partial derivative of the cost function with respect to θⱼ

For linear regression with MSE, the gradient is:

```
∂J(θ)/∂θⱼ = (1/m) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾ⱼ
```

In vector notation, the gradient can be written as:

```
∇J(θ) = (1/m) * X^T * (Xθ - y)
```

The update rule becomes:

```
θ := θ - α * (1/m) * X^T * (Xθ - y)
```

#### Normal Equation

The normal equation provides a closed-form solution to find the optimal parameters without iteration:

```
θ = (X^T * X)^(-1) * X^T * y
```

This method directly computes the values of θ that minimize the cost function by setting the gradient to zero and solving for θ.

## Implementation

Our implementation provides a flexible `LinearRegression` class that supports both gradient descent and normal equation optimization methods.

### Class Structure

```python
class OptimizationMethods(Enum):
  GRADIENT_DESCENT = 'gradient_descent'
  NORMAL_EQUATION = 'normal_equation'

class LinearRegression:
  def __init__(self, learning_rate: float=0.01, iter: int = 1000, 
               method: OptimizationMethods = OptimizationMethods.GRADIENT_DESCENT):
    # Initialize parameters
```

### Methods

1. **`__init__`**: Initializes the model with hyperparameters:
   - `learning_rate`: Step size for gradient descent (default: 0.01)
   - `iter`: Number of iterations for gradient descent (default: 1000)
   - `method`: Optimization method to use (default: GRADIENT_DESCENT)

2. **`_add_bias`**: Adds a column of ones to the feature matrix to account for the bias term.

3. **`fit`**: Trains the model on the provided data:
   - Adds bias term to features
   - Initializes weights
   - Calls the appropriate optimization method

4. **`_gradient_descent`**: Implements the gradient descent algorithm:
   - Iteratively updates weights based on the gradient of the cost function
   - Uses vectorized operations for efficiency

5. **`_normal_equation`**: Implements the normal equation method:
   - Directly computes the optimal weights using the closed-form solution

6. **`predict`**: Makes predictions on new data:
   - Adds bias term to features
   - Computes the dot product of features and weights

7. **`get_mse_loss`**: Calculates the Mean Squared Error between predictions and actual values.

### Usage Example

```python
import numpy as np
from LinearRegression.lin_reg import LinearRegression, OptimizationMethods

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(1000)

# Train the model using gradient descent
model_gd = LinearRegression(learning_rate=0.1, iter=1000, 
                           method=OptimizationMethods.GRADIENT_DESCENT)
model_gd.fit(X, y)
mse_gd = model_gd.get_mse_loss(X, y)
print(f"MSE with Gradient Descent: {mse_gd}")

# Train the model using normal equation
model_ne = LinearRegression(method=OptimizationMethods.NORMAL_EQUATION)
model_ne.fit(X, y)
mse_ne = model_ne.get_mse_loss(X, y)
print(f"MSE with Normal Equation: {mse_ne}")

# Make predictions on new data
X_new = np.array([[0], [2]])
predictions = model_gd.predict(X_new)
print(f"Predictions: {predictions}")
```

## Performance Considerations

1. **Gradient Descent vs. Normal Equation**:
   - Gradient Descent:
     - Advantages: Works well with large datasets, can be used with regularization
     - Disadvantages: Requires tuning of learning rate, may need many iterations
   - Normal Equation:
     - Advantages: No need to choose learning rate, no iterations needed
     - Disadvantages: Slow if number of features is large (O(n³) complexity for matrix inversion)

2. **Feature Scaling**:
   - Gradient descent converges faster when features are on a similar scale
   - Consider normalizing features (e.g., using StandardScaler) before training

3. **Learning Rate Selection**:
   - Too small: Slow convergence
   - Too large: May not converge or diverge
   - Consider using learning rate scheduling or adaptive methods

## Extensions

Possible extensions to this implementation include:

1. **Regularization**:
   - L1 regularization (Lasso)
   - L2 regularization (Ridge)
   - Elastic Net (combination of L1 and L2)

2. **Polynomial Features**:
   - Transform features to include polynomial terms
   - Allows modeling of non-linear relationships

3. **Stochastic Gradient Descent**:
   - Update weights using a single example at a time
   - Mini-batch gradient descent (update using a batch of examples)

4. **Early Stopping**:
   - Stop gradient descent when the improvement in cost function is below a threshold

5. **Learning Rate Scheduling**:
   - Decrease learning rate over time for better convergence

6. **Cross-Validation**:
   - Implement k-fold cross-validation for hyperparameter tuning
