# OpenVINO 介绍

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Architecture](https://img.shields.io/badge/arch-aarch64-red.svg)
![OS](https://img.shields.io/badge/OS-openEuler%2024.03%20LTS%20SP4-002FA7.svg)

## 简介

鲲鹏OpenVINO代码仓是针对开源 OpenVINO 项目在鲲鹏服务器上的适配与使能。

OpenVINO 是开源的深度学习推理优化框架，利用硬件特性加速 AI 模型，被广泛应用于目标检测、人脸识别等媒体相关推理场景。

## 目录结构

```text
├── docs/                                            # 文档
│   └── zh                                           # 中文文档
│       ├── installation_guide.md                    # 安装指南
│       ├── quick_start.md                           # 快速入门
│       └── LICENSE                                  # 文档许可证
├── examples
│   ├── facenet                                      # 人脸识别 FaceNet 示例   
│   │   ├── dataset                                  # LFW 数据集
│   │   ├── facenet_inference.py                     # 模型转换与推理
│   └── ultra-lightweight-face-detection-rfb-320     # 人脸检测示例
│       ├── test_face_detection_FRB320.py            # 模型转换与推理
│       └── test_face.jpg / face_result.jpg          # 检测图像/结果
├── LICENSE                                          # 开源许可证
└── README.md                                        # 项目说明文档
```

## 环境部署

鲲鹏OpenVINO的环境部署详见[安装指南](docs/zh/installation_guide.md)。

## 快速入门

### PyTorch 模型

使用 OpenVINO 将 PyTorch 模型转换并编译到 CPU，以及执行模型推理验证：

```python
import openvino as ov
import torch
import torchvision

# load PyTorch model into memory
model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", weights="DEFAULT")

# convert the model into OpenVINO model
example = torch.randn(1, 3, 224, 224)
ov_model = ov.convert_model(model, example_input=(example,))

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
output = compiled_model({0: example.numpy()})
```

### TensorFlow 模型

使用 OpenVINO 将 TensorFlow 模型转换并编译到 CPU，以及执行模型推理验证：

```python
import numpy as np
import openvino as ov
import tensorflow as tf

# load TensorFlow model into memory
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# convert the model into OpenVINO model
ov_model = ov.convert_model(model)

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
data = np.random.rand(1, 224, 224, 3)
output = compiled_model({0: data})
```

### 示例

本项目提供人脸检测和人脸识别两个 OpenVINO 推理示例，帮助用户快速验证鲲鹏服务器上的 OpenVINO 环境及模型推理能力。

* [人脸检测](examples/ultra-lightweight-face-detection-rfb-320/)：使用 `ultra-lightweight-face-detection-rfb-320` 模型进行人脸检测。

* [FaceNet 人脸识别](examples/facenet/)：使用 `facenet-20180408-102900` 模型提取人脸特征并进行人脸比对。

详细的模型准备及运行方法请参考 [examples/facenet](examples/facenet/README.md) 以及 [examples/ultra-lightweight-face-detection-rfb-320](examples/ultra-lightweight-face-detection-rfb-320/README.md)。

## 许可证

本项目采用Apache License Version 2.0许可证。详见[LICENSE](LICENSE)文件。
本项目的文档适用CC-BY 4.0许可证，具体请参见[docs/zh/LICENSE](docs/zh/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/openvino/issues)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/openvino/discussions)展开讨论交流。感谢您的支持。
