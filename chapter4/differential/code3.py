# 미분 계산 함수 나쁜 구현 예
def numerical_diff(f, x):
	h = 1e-50
	return (f(x + h) - f(x)) / h