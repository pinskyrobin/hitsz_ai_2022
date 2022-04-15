import logging
import threading

import numpy as np
import tensorflow as tf
from PIL import Image

from model_service.tfserving_model_service import TfServingBaseService
import os
logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
运行环境：tf2.1以上版本，python 3.6以上版本
modelArts AI引擎选择：TensorFlow tf2.1-python3.7
"""

class flower_service(TfServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.predict = None

        # 非阻塞方式加载saved_model模型，防止阻塞超时
        # self.label = {0: 'bee', 1: 'blackberry', 2: 'blanket', 3: 'bougainvillea', 4: 'bromelia',
        #                5: 'foxglove'}  # 创建图像标签列表
        self.label_list = ['bee', 'blackberry', 'blanket', 'bougainvillea', 'bromelia', 'foxglove']
        thread = threading.Thread(target=self.load_model)
        thread.start()

    def load_model(self):
        print('load model begin ======================================')
        print(os.listdir(self.model_path))  # 这个路径就是obs的路径，可以在日志界面上查看打印内容
        # load saved_model 格式的模型
        self.model = tf.saved_model.load(self.model_path)

        signature_defs = self.model.signatures.keys()
        signature = []
        # only one signature allowed
        for signature_def in signature_defs:
            signature.append(signature_def)

        if len(signature) == 1:
            model_signature = signature[0]
        else:
            logging.warning("signatures more than one, use serving_default signature from %s", signature)
            model_signature = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        self.predict = self.model.signatures[model_signature]

    def _preprocess(self, data):
        # 读取图片的二进制流转成数组，注意resize大小要跟训练时保持一致
        images = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                image1 = Image.open(file_content)
                image1 = np.array(image1, dtype=np.float32)
                image1.resize((100, 100, 3))
                images.append(image1)

        images = tf.convert_to_tensor(images, dtype=tf.dtypes.float32)
        preprocessed_data = images
        return preprocessed_data

    def _inference(self, data):
        return self.predict(data)

    def _postprocess(self, data):
        # 从模型的输出提取需要的数据，构造与config.json约定的输出格式
        outputs = {}
        logits = data["output_1"].numpy()[0].tolist()
        label_index = logits.index(max(logits))
        logits = ['%.4f' % logit for logit in logits]
        outputs['predicted_label'] = self.label_list[label_index]
        scores = dict(zip(self.label_list, logits))
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
        outputs['scores'] = scores
        return outputs
