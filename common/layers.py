import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import softmax, cross_entropy_error

# ReLU 활성화 함수 계층 구현
class Relu:
  def __init__(self):
    self.mask = None
    
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    
    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    
    return dx
  
# Sigmoid 활성화 함수 계층 구현
class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out
    
    return out
  
  def backward(sel, dout):
    dx = dout * (1.0 - self.out) * self.out
    
    return dx

# Affine 계층 구현
class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None
  
  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b
    
    return out
  
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)
    
    return dx

# Softmax-with-Loss 계층 구현
class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None # 손실
    self.y = None # softmax의 출력
    self.t = None # 정답 레이블(원-핫 벡터)
  
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    
    return self.loss
  
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size
    
    return dx