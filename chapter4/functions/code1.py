# 오차제곱합 구현
import numpy as np

# y는 신경망의 출력 값이 됩니다.
# t는 정답 레이블 값이 됩니다.
def sum_squares_error(y, t):
  return 0.5 * np.sum((y-t)**2)