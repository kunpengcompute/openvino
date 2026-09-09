import argparse
import cv2
import openvino as ov
import numpy as np

parser = argparse.ArgumentParser(
    description="Ultra-lightweight face detection (RFB-320) "
                "ONNX -> OpenVINO 推理脚本"
)

parser.add_argument(
    "--onnx",
    default=(
        "/workspace/open_model_zoo/public/"
        "ultra-lightweight-face-detection-rfb-320/"
        "ultra-lightweight-face-detection-rfb-320.onnx"
    ),
    help="ONNX 模型路径 (默认: 内置路径)",
)

parser.add_argument(
    "--image",
    default="/workspace/open_model_zoo/ovc_test/FRB320/face_test.jpg",
    help="输入图像路径 (默认: 内置测试图)",
)

parser.add_argument(
    "--output",
    default="face_result.jpg",
    help="输出图像路径 (默认: face_result.jpg)",
)

parser.add_argument(
    "--device",
    default="CPU",
    help="推理设备 (默认: CPU)",
)

parser.add_argument(
    "--confidence",
    type=float,
    default=0.5,
    help="置信度阈值 (默认: 0.5)",
)

parser.add_argument(
    "--precision",
    default="f16",
    choices=["f16", "f32"],
    help="推理精度提示 (默认: f16)",
)

args = parser.parse_args()


# --------------------------------------------------
# 1. 将 ONNX 转换为 OpenVINO 模型
#    注意：ov.convert_model() 不支持 compress_to_fp16 参数。
#    FP16 量化改在 compile_model 阶段通过 INFERENCE_PRECISION_HINT 实现。
# --------------------------------------------------

ov_model = ov.convert_model(
    args.onnx,
    input=[1, 3, 240, 320],
    output=["boxes", "scores"],
)


# --------------------------------------------------
# 2. 编译到指定设备
#    INFERENCE_PRECISION_HINT 等价于 --compress_to_fp16 True
# --------------------------------------------------

core = ov.Core()

compiled_model = core.compile_model(
    ov_model,
    args.device,
    config={"INFERENCE_PRECISION_HINT": args.precision},
)

input_layer = compiled_model.input(0)

output_boxes = compiled_model.output("boxes")
output_scores = compiled_model.output("scores")


# 确认转换后的模型信息
print("Input:")
print("  name :", input_layer.any_name)
print("  shape:", input_layer.shape)

print("Outputs:")

for out in compiled_model.outputs:
    print("  name :", out.any_name)
    print("  shape:", out.shape)


# --------------------------------------------------
# 3. 读取图像并预处理
# --------------------------------------------------

image = cv2.imread(args.image)

if image is None:
    raise RuntimeError(
        f"Failed to read image: {args.image}"
    )

original_h, original_w = image.shape[:2]

input_image = cv2.resize(image, (320, 240))

# HWC -> CHW（模型输入为 NCHW [1,3,240,320]）
input_image = input_image.transpose(2, 0, 1)

input_image = input_image[np.newaxis, ...]

input_image = input_image.astype(np.float32)

input_image = (input_image - 127.0) / 128.0


# --------------------------------------------------
# 4. 推理
# --------------------------------------------------

result = compiled_model(
    {input_layer: input_image}
)


boxes = result[output_boxes]
scores = result[output_scores]


# --------------------------------------------------
# 5. 后处理并绘制检测框
# --------------------------------------------------

num_anchors = scores.shape[1]

detected = 0

for i in range(num_anchors):

    confidence = scores[0, i, 1]

    if confidence > args.confidence:

        x_min, y_min, x_max, y_max = boxes[0, i]

        x_min = int(x_min * original_w)
        y_min = int(y_min * original_h)
        x_max = int(x_max * original_w)
        y_max = int(y_max * original_h)

        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        detected += 1


cv2.imwrite(args.output, image)

print(
    f"Detected {detected} face(s), "
    f"saved to {args.output}"
)
