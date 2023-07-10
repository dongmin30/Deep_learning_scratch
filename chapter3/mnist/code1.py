import sys, os
# 상위 폴더 dataset을 참조하기 위한 코드
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from dataset.mnist import load_mnist

# 처음 한 번은 몇 분 정도 걸린다.
(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape) # (60000, 784) 훈련이미지 : 60000개, 평탄화 1차원 배열 원소 : 784개
print(t_train.shape) # (60000, )
print(x_test.shape) # (10000, 784) 시험이미지 : 10000개, 평탄화 1차원 배열 원소 : 784개
print(t_test.shape) # (10000, )