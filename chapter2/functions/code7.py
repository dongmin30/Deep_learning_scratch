# ReLU 활성화 함수 구현해보기
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
  return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.1, 5.5)
plt.show()
