# facenet-20180408-102900

## 概述

facenet-20180408-102900 是基于 FaceNet（A Unified Embedding for Face Recognition and Clustering，人脸识别与聚类的统一嵌入）的人脸识别模型，源自 David Sandberg 的 [facenet](https://github.com/davidsandberg/facenet) 开源实现。模型输入对齐后的 160×160 人脸裁剪图，输出 512 维人脸特征向量（embedding），通过比较两张人脸 embedding 的余弦相似度判断是否为同一人。模型在 LFW 数据集上准确率达 99.14%。

原始模型为 TensorFlow 导出的 PB 文件。本仓库提供的测试用例完成 PB 文件到 OpenVINO IR 文件的完整转换，并基于 LFW 数据集进行推理，实现了同人或不同人脸的比对验证。

## 模型规格

| Metric           | Value       |
|------------------|-------------|
| Source framework | TensorFlow* |
| GFlops           | 2.846       |
| MParams          | 23.469      |
| LFW accuracy     | 99.14%      |

## 模型输入

模型接收单张对齐后的人脸裁剪图像（Image），具体规格如下：

| 属性 | 值 |
|------|------|
| 输入名称 | `input` |
| 输入类型 | Image（人脸裁剪图） |
| Shape | `1, 160, 160, 3` |
| 数据布局 | `B, H, W, C`（NHWC） |
| 数据类型 | `float32` |
| 取值范围 | 约 `[-1.0, 1.0]`（归一化后） |
| 颜色顺序 | `RGB` |

输入应为经过人脸检测（如 `ultra-lightweight-face-detection-rfb-320`）并裁剪出的人脸区域，而非整张原始图片。裁剪出的人脸框需 resize 到 160×160 后送入模型。

FaceNet 原始模型在 RGB 图像上训练，经 LFW 数据集实测，RGB 输入下同人 / 不同人的余弦相似度分离度优于 BGR。由于 `cv2.imread()` 读取的图像为 BGR 顺序，本示例预处理时通过 `cv2.cvtColor()` 转换为 RGB。

原始 TensorFlow 模型还包含 `batch_size`、`phase_train`、`batch_join:1:0` 等辅助输入，转换时已分别冻结为 `1`、`False`、`[0]`，部署模型仅保留上述单一图像输入。

## 模型输出

模型输出单张人脸的特征向量（embedding），具体规格如下：

| 属性 | 值 |
|------|------|
| 输出名称 | `embeddings` |
| Shape | `1, 512` |
| 数据类型 | `float32` |
| 输出含义 | 一张人脸的 512 维特征向量 |

该 512 维向量由 FaceNet 的 Inception-ResNet-v1 backbone 提取，是人脸在特征空间中的表示。同一人不同照片的 embedding 距离较近，不同人的 embedding 距离较远。输出本身不包含任何身份标签，需通过与底库向量比对完成识别。

## 运行示例

### 前置条件

已按 [安装指南](../../docs/zh/installation_guide.md) 完成 OpenVINO 环境部署，以及安装 Open Model Zoo。

下载 FaceNet 模型：
```bash
omz_downloader --name facenet-20180408-102900
```
下载 LFW 数据集：
```
mkdir -p /examples/facenet/dataset
cd /examples/facenet/dataset
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz
```

### 运行命令

```bash
# 自定义模型路径与验证数据集
python3 facenet_inference.py \
    --pb /path/to/20180408-102900.pb \
    --ir /path/to/facenet-20180408-102900.xml \
    --lfw /path/to/lfw \
    --device CPU \
    --num-people 5 \
    --num-images 2 \
    --threshold 0.42
```

### 参数说明

| 参数            | 默认值                                  | 说明                       |
|-----------------|----------------------------------------|----------------------------|
| `--pb`          | `/workspace/open_model_zoo/public/facenet-20180408-102900/20180408-102900/20180408-102900.pb` | TensorFlow PB 模型路径      |
| `--ir`          | `examples/facenet/facenet-20180408-102900.xml` | 输出的 OpenVINO IR xml 路径 |
| `--lfw`         | `examples/facenet/dataset/lfw`         | LFW 数据集根目录            |
| `--device`      | `CPU`                                  | 推理设备                    |
| `--num-people`  | `5`                                    | 从 LFW 选取的人数           |
| `--num-images`  | `2`                                    | 每人取几张图做同人比对       |
| `--threshold`   | `0.42`                                 | 余弦相似度阈值               |
| `--skip-convert`| -                 | 跳过转换，直接加载已有 IR    |


### 示例结果


LFW 数据集比对验证输出结果：

```text
[4/4] LFW validation results:
  People selected      : 5
  Same-person pairs    : 5
  Diff-person pairs    : 10
  Same sim mean/min/max: 0.7827 / 0.6705 / 0.9176
  Diff sim mean/min/max: 0.0038 / -0.2772 / 0.3814
  Threshold            : 0.42
  Same accepted        : 5/5
  Diff rejected        : 10/10
  Accuracy             : 15/15 = 100.00%
```

### benchmark_app 性能测试

`benchmark_app` 是 OpenVINO 自带的性能基准测试工具，用于测量模型在指定设备上的推理性能。它通过多次执行推理并统计延迟、吞吐量等指标，帮助评估模型部署后的实际运行效率。

测试运行命令如下：
```bash
/workspace/openvino/bin/aarch64/Release/benchmark_app \
    -m IR_file_path \
    -d CPU \
    -hint latency \
    -i image_path
```

## 合规信息

原始模型来自 [davidsandberg/facenet](https://github.com/davidsandberg/facenet)，遵循 MIT License。
