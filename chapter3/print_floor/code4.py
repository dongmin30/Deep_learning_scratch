# 소프트맥스 함수 오버플로 방지 작업 및 구현
import numpy as np

def softmax(a):
  
  c = np.max(a)
  exp_a = np.exp(a - c) # 오버플로 대책
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  
  return y