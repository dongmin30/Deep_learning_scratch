import numpy as np
import matplotlib.pylab as plt

def function_1(x):
	return 0.01*x**2 + 0.1*x

def numerical_diff(f, x):
	h = 1e-4
	return (f(x + h) - f(x - h)) / (2 * h)

x = np.arange(0.0, 20.0, 0.1) # 0에서 20까지 0.1 간격의 배열 x를 만든다.
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()