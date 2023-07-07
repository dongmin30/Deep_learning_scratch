# 시그모이드 함수 구현하기
import numpy as np
def sigmoid(x):
  return 1 / (1 + np.exp(-x))