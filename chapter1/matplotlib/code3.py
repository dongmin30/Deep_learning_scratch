import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('images/matplotlib_image.png')

plt.imshow(img)
plt.show()