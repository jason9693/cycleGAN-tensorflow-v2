from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.misc import imsave

img = imread('../img/test.jpg')
img = img / 255
print(img)

plt.imshow(img)
plt.show()

img = resize(img, (500,500))
print(img)

plt.imshow(img)
plt.show()

imsave('../img/gen.jpg',img)