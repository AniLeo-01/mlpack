import numpy as np
from enum import Enum

class OptimizationMethods(Enum):
  GRADIENT_DESCENT = 'gradient_descent'
  NORMAL_EQUATION = 'normal_equation'

class LinearRegression:
  def __init__(self, learning_rate: float=0.01, iter: int = 1000, method: OptimizationMethods = OptimizationMethods.GRADIENT_DESCENT, lambda_=0.1, regularization=None):
    self.learning_rate = learning_rate
    self.iter = iter
    self.method = method
    self._W = None
    self._b = None
    self.lambda_ = lambda_
    self.regularization = regularization

  def _add_bias(self, X):
    """Add a column of bias to the input matrix X"""
    return np.c_[np.ones(X.shape[0]), X]
  
  def fit(self, X, y):
    """Fit the model to the training data"""
    X = self._add_bias(X)
    # initialize the weights as random matrix
    self._W = np.random.randn(X.shape[1])
    if self.method == OptimizationMethods.GRADIENT_DESCENT:
      self._gradient_descent(X, y)
    elif self.method == OptimizationMethods.NORMAL_EQUATION:
      self._normal_equation(X,y)
    else:
      raise ValueError(f"Invalid optimization method: {self.method}")
    
  def _gradient_descent(self, X, y):
    """Gradient descent optimization"""
    for _ in range(self.iter):
      y_pred = X @ self._W
      grad = (X.T @ (y_pred - y))/X.shape[0]
      if self.regularization == 'lasso':
        #exclude the bias term and extend the weight vector with a zero to include the bias term
        grad += (self.lambda_/X.shape[0]) * np.r_[0, np.sign(self._W[1:])]
      elif self.regularization == 'ridge':
        grad += (self.lambda_/X.shape[0]) * np.r_[0, self._W[1:]]
      self._W -= self.learning_rate*grad

  def _normal_equation(self, X, y):
    """Normal equation optimization"""
    self._W = np.linalg.inv(X.T @ X) @ X.T @ y

  def predict(self, X):
    """Predict the target values"""
    X = self._add_bias(X)
    return X @ self._W
  
  def get_mse_loss(self, X, y):
    """Get the mean squared error loss"""
    y_pred = self.predict(X)
    return np.mean((y_pred - y)**2)
  
  def _standardize(self, X):
    """Standardize the input matrix X"""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
  
  def _lasso_regularization(self, X, y):
    """Lasso Regularization"""
    pass
  
  def _ridge_regularization(self, X, y):
    """Ridge Regularization"""
    pass

if __name__ == "__main__":
  np.random.seed(42)
  X = 2 * np.random.rand(1000, 1)
  y = 4 + 3 * X[:, 0] + np.random.randn(1000)
  # Train the model
  model = LinearRegression(learning_rate=0.1, iter=1000, method=OptimizationMethods.GRADIENT_DESCENT)
  model.fit(X, y)
  
  # Predictions
  predictions = model.predict(X)
  mse_loss = model.get_mse_loss(X, y)
  print(f"MSE Loss: {mse_loss}")