from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 文件来自网络，跟使用订阅算法生成的文件一致。如果自己训练的模型onnx转om格式失败，大概率是参数配置如NCHW配置不对

import os
import numpy as np
from PIL import Image
from hiai.nn_tensor_lib import NNTensor
from hiai.nntensor_list import NNTensorList
from model_service.hiai_model_service import HiaiBaseService

"""AIPP example
aipp_op {
    aipp_mode: static
    input_format : RGB888_U8

    mean_chn_0 : 123
    mean_chn_1 : 117
    mean_chn_2 : 104
}
"""

labels_list = []
if os.path.exists('labels.txt'):
  with open('labels.txt', 'r') as f:
    for line in f:
      if line.strip():
        labels_list.append(line.strip())


def keep_ratio_resize(im, base=100):
  short_side = min(float(im.size[0]), float(im.size[1]))
  resize_ratio = base / short_side
  resize_sides = int(round(resize_ratio * im.size[0])), int(round(resize_ratio * im.size[1]))
  im = im.resize(resize_sides)
  return im


def central_crop(im, base=100):
  width, height = im.size
  left = (width - base) / 2
  top = (height - base) / 2
  right = (width + base) / 2
  bottom = (height + base) / 2
  # Crop the center of the image
  im = im.crop((left, top, right, bottom))
  return im


class DemoService(HiaiBaseService):

  def _preprocess(self, data):

    preprocessed_data = {}
    images = []
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        image = keep_ratio_resize(image, base=100)
        image = central_crop(image, base=100)
        image = np.array(image)  # HWC
        image = image.transpose(2,0,1)
        # AIPP should use RGB format.
        # mean reg is applied in AIPP.
        # Transpose is applied in AIPP
        tensor = NNTensor(image)
        images.append(tensor)
    tensor_list = NNTensorList(images)
    preprocessed_data['images'] = tensor_list
    return preprocessed_data

  def _inference(self, data, image_info=None):
    result = {}
    for k, v in data.items():
      result[k] = self.model.proc(v)

    return result

  def _postprocess(self, data):
    outputs = {}
    prob = data['images'][0][0][0][0].tolist()
    outputs['scores'] = prob
    labels_list = ['bee', 'blackberry', 'blanket', 'bougainvillea', 'bromelia', 'foxglove']
    if labels_list:
      outputs['predicted_label'] = labels_list[int(np.argmax(prob))]
    else:
      outputs['predicted_label'] = str(int(np.argmax(prob)))

    return outputs
