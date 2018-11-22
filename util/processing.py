import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.io import imread
from skimage.transform import resize as imresize
import numpy as np

def show_images(images, file_path='generated/save.png'):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
        plt.imsave(file_path, img.reshape([sqrtimg,sqrtimg]))


    return fig

def mnist_preprocessing(img):
    return img * 2 -1

def mnist_deprocessing(logits):
    return (logits + 1) * 0.5

def load_image(filename, size=None, meta=False):
    """Load and resize an image from disk.
    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename) / 255
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        #new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)

    if meta:
        #print(img.shape)
        return[img, (img.shape[0],img.shape[1])]
    return img


# SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    return img  * 2 - 1
    #return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img):#, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    #img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    img = (img + 1) * 0.5
    # if rescale:
    #     vmin, vmax = img.min(), img.max()
    #     img = (img - vmin) / (vmax - vmin)
    return img

def shuffle_crop(img: np.array, desired_output: int):
    width = img.shape(0)
    height = img.shape(1)


    pass