import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from imageio import imread
import os
import glob

# Loading the images ======================================================
def load_data(basePath, img_size):
    """
    Get the x, y data for the images
    @param
        basePath: the base directory to find the structured flower images in
        img_size: dimension of the image (both x and y)
    
    @return
        x: a list of the image data, with shape (:, 150, 150, 3)
        y: the list of class labels associated with the corresponding image data
        keys: key for the string representation of y
    """

    keys, flowers = load_filepaths(basePath)
    x, y = load_images(flowers, img_size)
    # plot_images(x, y)
    return x, y, keys

def load_filepaths(basePath):
    """
    Get the file paths for every image
    @param
        basePath: the base directory to find the structured flower images in

    @return
        keys: the class labels
        filePaths: the list of file paths, for each key (2D list)
    """

    daisy = glob.glob(os.path.join(basePath + 'daisy', '*.jpg'))
    dandelion = glob.glob(os.path.join(basePath + 'dandelion', '*.jpg'))
    roses = glob.glob(os.path.join(basePath + 'roses', '*.jpg'))
    sunflowers = glob.glob(os.path.join(basePath + 'sunflowers', '*.jpg'))
    tulips = glob.glob(os.path.join(basePath + 'tulips', '*.jpg'))

    return ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'], [daisy, dandelion, roses, sunflowers, tulips]

def load_images(image_paths, img_size):
    """
    Load the given image file paths

    @param
        image_paths: 2D list of image file paths dimensions are: (class, file_path)
        img_size: dimension of the image (both x and y)

    @return
        x: a list of the image data, with shape (:, 150, 150, 3)
        y: the list of class labels associated with the corresponding image data
    """

    # round robbin to concatenate the arrays
    sizes = []
    for f_class in image_paths:
        sizes.append(len(f_class))

    x = []
    y = []
    for i in range(max(sizes)):
        for k, f_class in enumerate(image_paths):
            if i < len(f_class):
                # load image
                img = cv2.cvtColor(cv2.imread(f_class[i]), cv2.COLOR_BGR2RGB) / 255.0
                x_ = tf.image.resize(img, (img_size, img_size)).numpy()
                x.append(x_)
                y.append(k) # number for label
    
    return np.array(x), np.array(y)

def plot_images(x, y):
    """
    Plot the first 10 images, with their class labels
    
    @param
        x: image data
        y: class labels
    """
    fig = plt.figure(figsize=[15, 15])
    for i in range(10):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(x[i])
        ax.set_title(y[i])
# END loading the images ======================================================

if __name__ == "__main__":
    print("MAIN")