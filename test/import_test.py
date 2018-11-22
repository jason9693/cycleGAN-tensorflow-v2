import os
os.chdir(os.path.join(os.getcwd(),'../'))
print(os.getcwd())
from util import processing as util
import matplotlib.pyplot as plt

img = util.load_image('img/test.jpg', size=128)

#plt.plot(img)
img = util.preprocess_image(img)[None]
#print(img)
#print('min_val : {} . max_val : {}'.format(img.min(), img.max()))
#print('finish')
