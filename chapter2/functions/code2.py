# numpy 배열 자료형을 반환할수 있게 하기위한 계단 함수 구현
import numpy as np
def step_function(x):
  y = x > 0
  return y.astype(np.int)