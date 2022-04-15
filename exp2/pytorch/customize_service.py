from PIL import Image
import log
from model_service.pytorch_model_service import PTServingBaseService
from torchvision import models

import torch.nn as nn
import torch
import json

import numpy as np

logger = log.getLogger(__name__)

import torchvision.transforms as transforms

# 定义模型预处理
infer_transformation = transforms.Compose([
    transforms.Resize([100, 100], 3),
    # 需要处理成pytorch tensor
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[  # 图像归一化
        0.229, 0.224, 0.225])
])


import os


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.model = flowerModel(model_path)
        # print(os.listdir(os.path.dirname(model_path)+'./model')  # 模型所在路径
        # 加载标签
        self.label_list = ['bee', 'blackberry', 'blanket', 'bougainvillea', 'bromelia', 'foxglove']
        # 亦可通过文件标签文件加载
        # model目录下放置label.json文件，此处读取
        # dir_path = os.path.dirname(os.path.realpath(self.model_path))
        # with open(os.path.join(dir_path, 'label.json')) as f:
        #     self.label = json.load(f)

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            input_batch = []
            for file_name, file_content in v.items():
                with Image.open(file_content) as image1:
                    if torch.cuda.is_available():
                        input_batch.append(infer_transformation(image1).cuda())
                    else:
                        input_batch.append(infer_transformation(image1))
            input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
            print(input_batch_var.shape)
            preprocessed_data[k] = input_batch_var

        return preprocessed_data

    def _postprocess(self, data):
        results = {}
        for k, v in data.items():
            # result = torch.argmax(v[0])
            logits = v[0].tolist()
            label_index = logits.index(max(logits))
            logits = ['%.4f' % logit for logit in logits]
            results['predicted_label'] = self.label_list[label_index]
            scores = dict(zip(self.label_list, logits))
            scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
            results['scores'] = scores

        return results

    def _inference(self, data):
        """
        {'images': tensor([[ 0.0379, -0.1610, -0.1625, 0.1593, -0.0034, 0.1108]],grad_fn=)}
        """
        result = {}
        for k, v in data.items():
            result[k] = self.model(v)
        print(result)
        return result


def flowerModel(model_path, **kwargs):
    # 生成网络,因训练阶段只保存了模型参数，这里需要再定义一遍模型结构
    model = models.alexnet()
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 6),
    )

    # 加载模型
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
    # CPU或者GPU映射
    model.to(device)
    # 声明为推理模式
    model.eval()

    return model