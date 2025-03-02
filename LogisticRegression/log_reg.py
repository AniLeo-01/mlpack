import numpy as np
import random

class LogisticRegression:
  def __init__(self, lr = 0.01, iters = 100, regularization = None, lambda_=0.1):
    self.lr = lr
    self.iters = iters
    self._W = None
    self.regularization = regularization
    self.lambda_ = lambda_
    self.bias = None

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def fit(self, X, y):
    #initialize weights
    self._W = np.random.randn(X.shape[1])
    self.bias = random.random()
    
    for _ in range(self.iters):
      # calculate prediction
      y_pred = self._sigmoid(np.dot(X, self._W) + self.bias)

      # calculate gradient
      # dw = (xT(yi-y)/m)
      grad_w = (1/len(y)) * np.dot(X.T, (y_pred - y))

      # db = (1/m) * sum(yi-y)
      grad_b = (1/len(y)) * np.sum(y_pred - y)

      #l1 regularization
      if self.regularization == 'l1':
        grad_w += (self.lambda_/len(y)) * np.sign(self._W)
      
      #l2 regularization
      if self.regularization == 'l2':
        grad_w += (self.lambda_/len(y)) * self._W
      
      #update weights and bias
      self._W -= self.lr*grad_w
      self.bias -= self.lr*grad_b

  def predict_proba(self, X):
    linear_output = np.dot(X, self._W) + self.bias
    return self._sigmoid(linear_output)
  
  def predict(self, X):
    return np.where(self.predict_proba(X) >= 0.5, 1, 0)
  
  def _TP(self, y_pred, y):
    return np.sum((y_pred == 1) & (y == 1))
  
  def _TN(self, y_pred, y):
    return np.sum((y_pred == 0) & (y == 0))
  
  def _FP(self, y_pred, y):
    return np.sum((y_pred == 1) & (y == 0))
  
  def _FN(self, y_pred, y):
    return np.sum((y_pred == 0) & (y == 1))
  
  def accuracy_score(self, y_pred, y):
    TP = self._TP(y_pred, y)
    TN = self._TN(y_pred, y)
    FP = self._FP(y_pred, y)
    FN = self._FN(y_pred, y)
    return (TP + TN) / (TP + TN + FP + FN)
  
  def precision_score(self, y_pred, y):
    TP = self._TP(y_pred, y)
    FP = self._FP(y_pred, y)
    return TP / (TP + FP)
  
  def recall_score(self, y_pred, y):
    TP = self._TP(y_pred, y)
    FN = self._FN(y_pred, y)
    return TP / (TP + FN)
      
if __name__ == "__main__":
  X = np.array([[1, 2], [3, 4], [5, 6]])
  y = np.array([0, 1, 1])
  model = LogisticRegression()
  model.fit(X, y)
  print(model.predict_proba(X))
  print(model.predict(X))
  print(model.accuracy_score(model.predict(X), y))