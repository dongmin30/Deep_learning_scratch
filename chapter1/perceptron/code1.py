# numpy, 편향, 가중치 개념 없이 AND 게이트 구현
def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  if tmp > theta:
    return 1
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))