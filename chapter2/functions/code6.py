# 계단 함수 그래프와 시그모이드 함수 그래프 비교
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
  return np.array(x > 0, dtype=np.int32)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x1 = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x1)
y2 = sigmoid(x1)
plt.plot(x1, y1, linestyle="--")
plt.plot(x1, y2)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()