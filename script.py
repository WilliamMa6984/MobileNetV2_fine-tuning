import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import os
import glob

from tensorflow import keras
from tensorflow.keras import layers

# import evaluation metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from script import *

from keras.applications.mobilenet_v2 import preprocess_input

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
    fig.set_facecolor('white')
    for i in range(10):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(x[i])
        ax.set_title(y[i])
# END loading the images ======================================================

# Compile and fine tune model ======================================================
def compile_and_tune(x, y, lr, mmtm, img_size, epochs):
    """
    Initialise the MobileNetV2 model using imagenet, replace with a new output dense layer, compile the new model, and train it.
    Freezes up to the 7th last layer (trains the last dense, and 2 2Dconvolutional layers)

    @param
        x: x-training data
        y: ground truth of training data (y-train)
        lr: learning rate
        mmtm: momentum
        img_size: size of input images (should be square ratio)

    @return
        model: the new, fine tuned model
        history: the history of the model's training
    """

    img_dim = (img_size, img_size, 3)

    # using imagenet weights - pretrained model
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=img_dim,
        alpha=1.0, # control width of network layers (less than 1.0 means decreases no. filters, more than 1.0 means increases no. filters)
        include_top=True, # include fully connected layer
        input_tensor=None, # none
        pooling=None, # maybe we can edit this?
        # classes=5, # default was 1000, maybe we dont need to replace last layer with dense layer of 5 if this is already 5?
        classifier_activation='softmax' # default
    )

    # connect the second last layer to the new dense layer, replacing the original last dense layer
    outputs = layers.Dense(5)(model.layers[-2].output)
    new_model = keras.Model(inputs=model.input, outputs=outputs)
    
    for layer in new_model.layers[:-7]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr, 
        momentum=mmtm,
        nesterov=False, 
        name="SGD"
    )

    new_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=['accuracy'])

    history = new_model.fit(x, y,
            batch_size=50,
            epochs=epochs,
            validation_split=0.2,
            validation_batch_size=50
        )
    
    return new_model, history

def eval_model(model, x_train, y_train, x_test, y_test, history):
    """
    Evaluates the performance of the training and testing set using the provided predictor.
    Code adapted from CAB420 lectures/practicals.

    @param
        model: the predictor
        x_train: the training features data
        y_train: the training ground truths
        x_test: the testing features data
        y_test: the testing ground truths
        history: the history of accuracy and loss over the epochs during the training of the neural network model
    """

    fig = plt.figure(figsize=[21, 6])
    fig.set_facecolor('white')
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.legend()
    ax.set_title('Training Loss/Accuracy')

    ax = fig.add_subplot(1, 3, 2)
    pred = model.predict(x_train)
    best_guesses = tf.argmax(pred, axis=1)
    confusion_mat = ConfusionMatrixDisplay(confusion_matrix(y_train, best_guesses), display_labels=range(5))
    confusion_mat.plot(ax = ax)
    ax.set_title('Training Accuracy: ' + str(sum(best_guesses.numpy() == y_train)/len(y_train)))
    
    ax = fig.add_subplot(1, 3, 3)
    pred = model.predict(x_test)
    best_guesses = tf.argmax(pred, axis=1)
    confusion_mat = ConfusionMatrixDisplay(confusion_matrix(y_test, best_guesses), display_labels=range(5))
    confusion_mat.plot(ax = ax)    
    ax.set_title('Testing Accuracy: ' + str(sum(best_guesses.numpy() == y_test)/len(y_test)))

    print(classification_report(y_test, best_guesses))

# END Compile and fine tune model ======================================================

if __name__ == "__main__":
    img_size = 96

    basePath = "C:\\Users\\n10491694\\Downloads\\small_flower_dataset\\" # replace with correct path

    x, y, labels = load_data(basePath, img_size)
    # plot_images(x, y)
    
    train_X = x[:750]
    train_y = y[:750]

    test_X = x[750:]
    test_y = y[750:]

    lr = 0.01
    mmtm = 0
    model, history = compile_and_tune(train_X, train_y, lr, mmtm, img_size, 10)
    eval_model(model, train_X, train_y, test_X, test_y, history)
    plt.show()

    lr = 0.001
    mmtm = 0
    model, history = compile_and_tune(train_X, train_y, lr, mmtm, img_size, 40)
    eval_model(model, train_X, train_y, test_X, test_y, history)
    plt.show()

    lr = 0.0001
    mmtm = 0
    model, history = compile_and_tune(train_X, train_y, lr, mmtm, img_size, 100)
    eval_model(model, train_X, train_y, test_X, test_y, history)
    plt.show()

    # 1: accuracy 0.
    # 2: accuracy 0.
    # 3: accuracy 0.

    print("MAIN")