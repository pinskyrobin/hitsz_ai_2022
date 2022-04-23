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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping

# If following packages cannot be recognized,
# please use please use "tensorflow.keras.optimizer" instead
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
# from tensorflow.python.keras.optimizer.adam import Adam
# from tensorflow.python.keras.optimizer.gradient_descent import SGD

# Fix prediction failure in macOS environment
# If this line raise any error, please just COMMENT it
tf.executing_eagerly()


# All hyper-parameters used in this program are listed in class Parameters
# You can refer any of them by using "_.xxx"
class Parameters:
    def __init__(self):
        cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.path_train = os.path.join(cur_dir, 'flower_photos/')
        self.path_test = os.path.join(cur_dir, 'TestImages/')
        self.wide = 224
        self.height = 224
        self.channel = 3
        self.lr = 0.001
        self.epochs = 20
        self.dropout = 0.5
        self.batch_size = 64
        self.split = 0.3
        self.seed = 42

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

    (x_train, x_val, y_train, y_val) = train_test_split(
        data, label, test_size=_.split, random_state=_.seed)

    # Standardize the data
    x_train = x_train / 255
    x_val = x_val / 255

    return x_train, x_val, y_train, y_val


def build():
    model = Sequential()
    return model


def train(model, x_train, y_train, x_val, y_val):

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

    opt = Adam(learning_rate=_.lr)
    # sgd = SGD(lr=_.lr, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=_.epochs,
                        callbacks=[es],
                        validation_data=(x_val, y_val),
                        batch_size=_.batch_size,
                        verbose=2)

    model.summary()

    model.save_weights('flower_model.pkl')

    # Draw the train and validation loss
    acc = history.history['acc']
    val_acc = history.history['val_acc']
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
        print("====" + im + "====")
        img = cv2.imread(im)
        img = cv2.resize(img, (_.wide, _.height))
        imgs.append(img)
    imgs = np.asarray(imgs, np.float32)
    print("shape of data:", imgs.shape)

    prediction = np.argmax(model.predict(imgs), axis=1)

    # Draw the prediction
    for i in range(np.size(prediction)):
        print("第", i + 1, "朵花预测:" + _.flower_dict[prediction[i]])
        img = plt.imread(_.path_test + "test" + str(i + 1) +
                         ".jpg")
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    x_train, x_val, y_train, y_val = preprocess()

    model = build()

    model = train(model, x_train, y_train, x_val, y_val)

    predict(model)
