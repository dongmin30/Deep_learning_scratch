# 교차 엔트로피 오차 함수 구현
import numpy as np

# y는 신경망의 출력 값이 됩니다.
# t는 정답 레이블 값이 됩니다.
def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))