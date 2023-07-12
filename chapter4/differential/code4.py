# 미분 계산 개선점 반영
def numerical_diff(f, x):
  # 반올림 오차가 발생하지 않도록 수치 개선
	h = 1e-4 # 0.0001
  # 수치 미분의 오차를 줄이기 위해 함수 f의 차분 계산
  # 이를 중심 차분 혹은 중앙 차분이라고 표현
	return (f(x + h) - f(x - h)) / (2 * h)