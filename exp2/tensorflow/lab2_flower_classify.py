import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

###############################################################################
# pip install packages are listed in requirements.txt #
###############################################################################

# If following packages cannot be recognized,
# please use "tensorflow.keras.xxx" instead

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from keras_preprocessing.image import ImageDataGenerator

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator


# If following packages cannot be recognized,
# please use please use "tensorflow.keras.optimizers" instead

# from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.adam import Adam


# from tensorflow.keras.optimizers import Adam

# Fix prediction failure in macOS environment
# If this line raise any error, please just COMMENT it

# tf.executing_eagerly()


# All hyper-parameters used in this program are listed in class Parameters
# You can refer any of them by using "_.xxx"
class Parameters:
    def __init__(self):
        cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.path_train = os.path.join(cur_dir, 'flower_photos/')
        self.path_test = os.path.join(cur_dir, 'TestImages/')
        self.wide = 100
        self.height = 100
        self.channel = 3
        self.lr = 0.001
        self.epochs = 90
        self.batch_size = 32
        self.split = 0.20
        self.verbose = 2

        self.flower_dict = {0: 'bee', 1: 'blackberry', 2: 'blanket',
                            3: 'bougainvillea', 4: 'bromelia', 5: 'foxglove'}


_ = Parameters()


# API for reading image from disk
def read_img(path, wide, height):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []

    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s'%(im))
            img = cv2.imread(im)
            img = cv2.resize(img, (wide, height))
            imgs.append(img)
            labels.append(idx)

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def preprocess():
    data, label = read_img(_.path_train, _.wide, _.height)
    print("shape of data:", data.shape)
    print("shape of label:", label.shape)

    seed = 109
    np.random.seed(seed)

    (x_train, x_val, y_train, y_val) = train_test_split(
        data, label, test_size=_.split, random_state=seed)

    # Standardize the data
    x_val = x_val / 255

    return x_train, x_val, y_train, y_val


def build():
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(6, activation="softmax")
    ])

    # opt = SGD(learning_rate=_.lr, momentum=0.9)
    opt = Adam(learning_rate=_.lr)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train(model, x_train, y_train, x_val, y_val):

    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    image_gen_train.fit(x_train)

    history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=_.batch_size),
                        validation_data=(x_val, y_val),
                        epochs=_.epochs,
                        steps_per_epoch=len(x_train) // _.batch_size,
                        verbose=_.verbose,
                        shuffle=True)

    model.summary()

    model.save('flower_model.h5')

    # Draw the train and validation loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    return model


def predict(model):
    imgs = []

    for im in glob.glob(_.path_test + '/*.jpg'):
        print('reading the images:%s' % (im))
        img = cv2.imread(im)
        img = cv2.resize(img, (_.wide, _.height))
        imgs.append(img)
    imgs = np.asarray(imgs, np.float32)
    print("shape of data:", imgs.shape)

    prediction = np.argmax(model.predict(imgs), axis=1)

    # Draw the prediction
    for i in range(np.size(prediction)):
        print("第", i + 1, "朵花预测:", prediction[i])
        img = plt.imread(_.path_test + "test" + str(i + 1) + ".jpg")
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    x_train, x_val, y_train, y_val = preprocess()

    model = build()

    model = train(model, x_train, y_train, x_val, y_val)

    predict(model)
