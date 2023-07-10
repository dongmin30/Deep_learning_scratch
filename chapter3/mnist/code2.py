import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 레이블이 갖는 값 출력
img = x_train[0]
label = t_train[0]
print(label)  # 5

# 이미지 출력 라인
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)