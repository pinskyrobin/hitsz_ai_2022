import glob                # 用于查询符合特定规则的文件路径名
import os                  # 处理文件和目录
import cv2                 # 用于图像处理
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential  #从tensorflow.keras模块下导入layers，optimizers, datasets, Sequential等方法
import numpy as np                #导入numpy数据库
import matplotlib.pyplot as plt   #导入matplotlib.pyplot模块，主要用于展示图像
from sklearn.model_selection import train_test_split   #从sklearn.model_selection模块导入train_test_split方法，用于拆分数据集

# 所用第三方包的安装方式，安装前更新pip:
# python -m pip install --upgrade pip
# pip install opencv-python # 对应cv2
# pip install tensorflow
# pip install matplotlib
# pip install sklearn

path = './flower_photos/'   # 数据集的相对地址，改为你自己的，建议将数据集放入代码文件夹下

# Todo 对图片进行缩放，统一处理为大小为w*h的图像，具体数据需自己定
w = 10       #设置图片宽度
h = 10       #设置图片高度
c = 3        #设置图片通道为3


def read_img(path):                                                    # 定义函数read_img，用于读取图像数据，并且对图像进行resize格式统一处理
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]  # 创建层级列表cate，用于对数据存放目录下面的数据文件夹进行遍历，os.path.isdir用于判断文件是否是目录，然后对是目录的文件(os.listdir(path))进行遍历
    imgs=[]                                                            # 创建保存图像的空列表
    labels=[]                                                          # 创建用于保存图像标签的空列表
    for idx,folder in enumerate(cate):                                # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和下标,一般用在for循环当中
        for im in glob.glob(folder+'/*.jpg'):                         # 利用glob.glob函数搜索每个层级文件下面符合特定格式“/*.jpg”的图片，并进行遍历
            #print('reading the images:%s'%(im))                      # 遍历图像的同时，打印每张图片的“路径+名称”信息
            img=cv2.imread(im)                                        # 利用cv2.imread函数读取每一张被遍历的图像并将其赋值给img
            img=cv2.resize(img,(w,h))                                 # 利用cv2.resize函数对每张img图像进行大小缩放，统一处理为大小为w*h的图像
            imgs.append(img)                                          # 将每张经过处理的图像数据保存在之前创建的imgs空列表当中
            labels.append(idx)                                        # 将每张经过处理的图像的标签数据保存在之前创建的labels列表当中
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)   # 利用np.asarray函数对生成的imgs和labels列表数据进行转化，之后转化成数组数据（imgs转成浮点数型，labels转成整数型）


data, label = read_img(path)                                           # 将read_img函数处理之后的数据定义为样本数据data和标签数据label
print("shape of data:",data.shape)                                     # 查看样本数据的大小
print("shape of label:",label.shape)                                   # 查看标签数据的大小


seed = 109             # 设置随机数种子，即seed值
np.random.seed(seed)   # 保证生成的随机数具有可预测性,即相同的种子（seed值）所产生的随机数是相同的

(x_train, x_val, y_train, y_val) = train_test_split(data, label, test_size=0.20, random_state=seed) #拆分数据集一部分为训练集一部分为验证集，拆分比例可调整
x_train = x_train / 255  #训练集图片标准化
x_val = x_val / 255      #测试集图片标准化

flower_dict = {0:'bee',1:'blackberry',2:'blanket',3:'bougainvillea',4:'bromelia',5:'foxglove'} #创建图像标签列表

# Todo 自行实现模型结构
model = Sequential([])

# Todo 可调整超参数lr，可选择其他优化器
opt = optimizers.Adam(learning_rate=0.0001)   #使用Adam优化器，优化模型参数。lr(learning rate, 学习率)

#编译模型以供训练。使用多分类损失函数'sparse_categorical_crossentropy'，使用metrics=['accuracy']即评估模型在训练和测试时的性能的指标，使用的准确率。
# Todo可选择其他损失函数
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Todo 可调整超参数
#训练模型，决定训练集和验证集，batch size：进行梯度下降训练模型时每个batch包含的样本数。
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val),batch_size=200, verbose=2)
#输出模型的结构和参数量
model.summary()

path_test = './TestImages/' # 测试图像的地址 （改为自己的）

imgs=[]                                                 # 创建保存图像的空列表
for im in glob.glob(path_test+'/*.jpg'):               # 利用glob.glob函数搜索每个层级文件下面符合特定格式“/*.jpg”进行遍历
    #print('reading the images:%s'%(im))                # 遍历图像的同时，打印每张图片的“路径+名称”信息
    img=cv2.imread(im)                                  # 利用io.imread函数读取每一张被遍历的图像并将其赋值给img
    img=cv2.resize(img,(w,h))                           # 利用cv2.resize函数对每张img图像进行大小缩放，统一处理为大小为w*h的图像
    imgs.append(img)                                    # 将每张经过处理的图像数据保存在之前创建的imgs空列表当中
imgs = np.asarray(imgs,np.float32)                      # 利用np.asarray()函数对imgs进行数据转换
print("shape of data:",imgs.shape)


prediction = np.argmax(model.predict(imgs), axis=1) #将图像导入模型进行预测
# prediction = model.predict_classes(imgs)      #将图像导入模型进行预测，旧版本tf可以使用该接口，最新版本已废弃
#绘制预测图像
for i in range(np.size(prediction)):
    #打印每张图像的预测结果
    print("第",i+1,"朵花预测:"+flower_dict[prediction[i]])  # flower_dict:定义的标签列表，prediction[i]：预测的结果
    img = plt.imread(path_test+"test"+str(i+1)+".jpg")      # 使用imread()函数读入对应的图片
    #img = plt.imread(path_test)
    plt.imshow(img)              #展示图片
    plt.show()                   #显示图片