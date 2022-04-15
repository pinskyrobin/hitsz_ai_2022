import glob                # 用于查询符合特定规则的文件路径名
import os                  # 处理文件和目录
import cv2                 # 用于图像处理
import tensorflow as tf
# 从tensorflow.keras模块下导入layers，optimizers, datasets, Sequential等方法
from tensorflow.keras import layers, optimizers, datasets, Sequential
import numpy as np  # 导入numpy数据库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，主要用于展示图像
from sklearn.model_selection import train_test_split
from torch import eig  # 从sklearn.model_selection模块导入train_test_split方法，用于拆分数据集

# 所用第三方包的安装方式，安装前更新pip:
# python -m pip install --upgrade pip
# pip install opencv-python # 对应cv2
# pip install tensorflow
# pip install matplotlib
# pip install sklearn

class params:
    def __init__(self):
        cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.path_train = os.path.join(cur_dir, 'flower_photos/')
        self.path_test = os.path.join(cur_dir, 'TestImages/')
        self.wide = 128
        self.height = 128
        self.channel = 3
        self.lr = 0.0001
        self.epochs = 100
        self.batch_size = 64
        self.split = 0.2

_ = params()

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

# Read and process data
def preprocess(path_train, wide, height, split=0.2):
    data, label = read_img(path_train, wide, height)
    print("shape of data:", data.shape)
    print("shape of label:", label.shape)

    seed = 109
    np.random.seed(seed)

    (x_train, x_val, y_train, y_val) = train_test_split(
        data, label, test_size=split, random_state=seed)

    # Standardize the data
    x_train = x_train / 255
    x_val = x_val / 255

    flower_dict = {0: 'bee', 1: 'blackberry', 2: 'blanket',
                   3: 'bougainvillea', 4: 'bromelia', 5: 'foxglove'}

    return x_train, x_val, y_train, y_val, flower_dict


def build():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
              input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    return model


def train(model, x_train, y_train, x_val, y_val, lr, epochs, batch_size):

    opt = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        batch_size=batch_size,
                        verbose=2)

    model.summary()

    model.save('flower_model.h5')

    # draw the train and validation loss
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


def predict(model, path_test, wide, height, flower_dict):
    imgs = []
    for im in glob.glob(path_test + '/*.jpg'):
        img = cv2.imread(im)
        img = cv2.resize(img, (wide, height))
        imgs.append(img)
    imgs = np.asarray(imgs, np.float32)
    print("shape of data:", imgs.shape)

    prediction = np.argmax(model.predict(imgs), axis=1)
    
    # draw the prediction
    for i in range(np.size(prediction)):
        print("第", i+1, "朵花预测:"+flower_dict[prediction[i]])
        img = plt.imread(path_test+"test"+str(i+1) +
                         ".jpg")
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    
    x_train, x_val, y_train, y_val, flower_dict = preprocess(
        _.path_train, _.wide, _.height, _.split)

    model = build()

    model = train(model, x_train, y_train, x_val,
                  y_val, _.lr, _.epochs, _.batch_size)

    predict(model, _.path_test, _.wide, _.height, flower_dict)
