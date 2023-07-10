# 소프트맥스 함수 구현
import numpy as np

def softmax(a):
  exp_a = np.exp(a) # 지수 함수
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  
  return y