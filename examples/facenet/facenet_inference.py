#!/usr/bin/env python3
"""
FaceNet TensorFlow PB -> OpenVINO IR 转换 + LFW 推理验证
合并自 convert_facenet.py 和 facenet_inference.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import openvino as ov
from openvino import opset13 as ops

parser = argparse.ArgumentParser(
    description="FaceNet PB -> OpenVINO IR + LFW 推理验证"
)

parser.add_argument(
    "--pb",
    default=(
        "/workspace/open_model_zoo/public/"
        "facenet-20180408-102900/"
        "20180408-102900/"
        "20180408-102900.pb"
    ),
    help="TensorFlow PB 模型路径",
)

parser.add_argument(
    "--ir",
    default=(
        "/workspace/open_model_zoo/ovc_test/facenet/"
        "facenet-20180408-102900.xml"
    ),
    help="输出的 OpenVINO IR xml 路径",
)

parser.add_argument(
    "--lfw",
    default=(
        "/workspace/open_model_zoo/ovc_test/facenet/"
        "dataset/lfw"
    ),
    help="LFW 数据集根目录",
)

parser.add_argument(
    "--device",
    default="CPU",
    help="推理设备 (默认: CPU)",
)

parser.add_argument(
    "--num-people",
    type=int,
    default=5,
    help="从 LFW 选取的人数 (默认: 5)",
)

parser.add_argument(
    "--num-images",
    type=int,
    default=2,
    help="每人取几张图做同人比对 (默认: 2)",
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.42,
    help=" cosine 相似度阈值 (默认: 0.42)",
)

parser.add_argument(
    "--skip-convert",
    action="store_true",
    help="跳过转换，直接加载已有 IR",
)

args = parser.parse_args()


# ============================================================
# 辅助函数
# ============================================================

def find_input(model, name):
    for inp in model.inputs:
        if inp.get_any_name() == name:
            return inp
    return None


def find_output(model, name):
    for out in model.outputs:
        if out.get_any_name() == name:
            return out
    return None


def freeze_input_to_constant(inp, value):
    """将输入 Parameter 的所有消费者替换为 Constant"""
    const = ops.constant(value, inp.element_type)

    for consumer in list(inp.get_target_inputs()):
        consumer.replace_source_output(const.output(0))


# ============================================================
# 1. 转换：TensorFlow PB -> OpenVINO IR
# ============================================================

def convert_pb_to_ir(pb_path, ir_path):
    """
    完整流程：
      - PB -> OpenVINO Model
      - 静态化 batch=1
      - 冻结辅助输入 (batch_size, phase_train, batch_join:1:0)
      - 构造干净模型 (仅保留 image_input -> embeddings)
      - 保存 FP16 IR
    """
    if not os.path.isfile(pb_path):
        print(f"[ERROR] PB file not found: {pb_path}")
        sys.exit(1)

    print(f"[1/4] Converting PB -> OpenVINO: {pb_path}")

    model = ov.convert_model(pb_path, verbose=False)

    # 定位所需张量
    batch_size = find_input(model, "batch_size:0")
    phase_train = find_input(model, "phase_train:0")
    image_input = find_input(model, "batch_join:0:0")
    batch_join_aux = find_input(model, "batch_join:1:0")
    embedding_output = find_output(model, "embeddings:0")

    if image_input is None or embedding_output is None:
        print("[ERROR] Required tensors not found")
        sys.exit(1)

    # 静态化 batch = 1
    reshape_map = {"batch_join:0:0": [1, 160, 160, 3]}

    if batch_size is not None:
        reshape_map["batch_size:0"] = [1]

    if batch_join_aux is not None:
        reshape_map["batch_join:1:0"] = [1]

    model.reshape(reshape_map)

    # 冻结 TensorFlow 辅助输入
    if batch_size is not None:
        freeze_input_to_constant(batch_size, [1])

    if phase_train is not None:
        freeze_input_to_constant(phase_train, False)

    if batch_join_aux is not None:
        freeze_input_to_constant(batch_join_aux, [0])

    # 重新查找输出（reshape 后引用可能失效）
    embedding_output = find_output(model, "embeddings:0")

    # 构造干净部署模型：仅 1 输入 1 输出
    image_parameter = image_input.get_node()

    clean_model = ov.Model(
        [embedding_output],
        [image_parameter],
        "facenet"
    )

    image_parameter.set_friendly_name("input")
    embedding_output.get_node().set_friendly_name("embeddings")

    # 验证
    actual_in = list(
        clean_model.inputs[0].partial_shape.to_shape()
    )

    actual_out = list(
        clean_model.outputs[0].partial_shape.to_shape()
    )

    expected_in = [1, 160, 160, 3]
    expected_out = [1, 512]

    if actual_in != expected_in or actual_out != expected_out:
        print(
            f"[ERROR] Shape mismatch: "
            f"in={actual_in} (expect {expected_in}), "
            f"out={actual_out} (expect {expected_out})"
        )
        sys.exit(1)

    if len(clean_model.inputs) != 1 or len(clean_model.outputs) != 1:
        print("[ERROR] Model should have exactly 1 input and 1 output")
        sys.exit(1)

    # 保存 FP16 IR
    os.makedirs(os.path.dirname(ir_path), exist_ok=True)

    ov.save_model(
        clean_model,
        ir_path,
        compress_to_fp16=True
    )

    print(f"[2/4] IR saved (FP16): {ir_path}")

    return ir_path


# ============================================================
# 2. 推理辅助
# ============================================================

def load_compiled_model(ir_path, device):
    """加载 IR 并编译到指定设备"""
    core = ov.Core()

    model = core.read_model(ir_path)

    compiled = core.compile_model(model, device)

    return compiled, compiled.input(0), compiled.output(0)


def get_embedding(image_path, compiled_model, input_layer, output_layer):
    """
    预处理流程（与原版 FaceNet 一致）：
      - resize 160x160
      - BGR -> RGB（FaceNet 在 RGB 上训练，LFW 实测优于 BGR）
      - (image - 127.5) / 128.0
      - NHWC 布局
      - L2 归一化
    """
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {image_path}"
        )

    image = cv2.resize(image, (160, 160))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_image = image.astype(np.float32)

    input_image = (input_image - 127.5) / 128.0

    input_image = input_image[np.newaxis, ...]

    result = compiled_model({input_layer: input_image})

    embedding = result[output_layer][0]

    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding


def cosine_similarity(emb1, emb2):
    """已 L2 归一化向量的点积即余弦相似度"""
    return float(np.dot(emb1, emb2))


# ============================================================
# 3. LFW 验证
# ============================================================

def run_lfw_test(
    compiled_model,
    input_layer,
    output_layer,
    lfw_root,
    num_people,
    num_images_per_person,
    threshold
):
    """从 LFW 选取样本做人脸比对验证"""
    people = sorted(os.listdir(lfw_root))

    selected = []

    for person in people:
        person_dir = os.path.join(lfw_root, person)

        if not os.path.isdir(person_dir):
            continue

        images = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])

        if len(images) >= num_images_per_person:
            selected.append((person, images))

        if len(selected) >= num_people:
            break

    if not selected:
        print(f"[ERROR] No valid people found in: {lfw_root}")
        sys.exit(1)

    # 提取所有 embedding
    embeddings = {}

    for person, images in selected:
        for img_name in images[:num_images_per_person]:
            img_path = os.path.join(lfw_root, person, img_name)

            emb = get_embedding(
                img_path,
                compiled_model,
                input_layer,
                output_layer
            )

            embeddings[(person, img_name)] = emb

    # 同人对
    same_sims = []

    for person, images in selected:
        imgs = images[:num_images_per_person]

        for i in range(len(imgs) - 1):
            sim = cosine_similarity(
                embeddings[(person, imgs[i])],
                embeddings[(person, imgs[i + 1])]
            )

            same_sims.append(sim)

    # 不同人对（第一张图两两组合）
    diff_sims = []

    first_images = [
        (person, images[0])
        for person, images in selected
    ]

    for i in range(len(first_images)):
        for j in range(i + 1, len(first_images)):
            p1, img1 = first_images[i]
            p2, img2 = first_images[j]

            sim = cosine_similarity(
                embeddings[(p1, img1)],
                embeddings[(p2, img2)]
            )

            diff_sims.append(sim)

    same_sims = np.array(same_sims)
    diff_sims = np.array(diff_sims)

    # 验证结论
    total = len(same_sims) + len(diff_sims)
    same_correct = int((same_sims >= threshold).sum())
    diff_correct = int((diff_sims < threshold).sum())
    correct = same_correct + diff_correct

    print(f"[4/4] LFW validation results:")
    print(f"  People selected      : {len(selected)}")
    print(f"  Same-person pairs    : {len(same_sims)}")
    print(f"  Diff-person pairs    : {len(diff_sims)}")
    print(f"  Same sim mean/min/max: "
          f"{same_sims.mean():.4f} / "
          f"{same_sims.min():.4f} / "
          f"{same_sims.max():.4f}")
    print(f"  Diff sim mean/min/max: "
          f"{diff_sims.mean():.4f} / "
          f"{diff_sims.min():.4f} / "
          f"{diff_sims.max():.4f}")
    print(f"  Threshold            : {threshold}")
    print(f"  Same accepted        : {same_correct}/{len(same_sims)}")
    print(f"  Diff rejected        : {diff_correct}/{len(diff_sims)}")
    print(f"  Accuracy             : {correct}/{total} = {correct/total:.2%}")


# ============================================================
# Main
# ============================================================

def main():
    # 步骤 1: 转换（可跳过）
    if args.skip_convert:
        if not os.path.isfile(args.ir):
            print(f"[ERROR] IR not found: {args.ir}")
            sys.exit(1)

        print(f"[1/4] Skip convert, use existing IR: {args.ir}")
    else:
        convert_pb_to_ir(args.pb, args.ir)

    # 步骤 2: 加载并编译
    print(f"[3/4] Loading and compiling to {args.device}...")

    compiled_model, input_layer, output_layer = load_compiled_model(
        args.ir, args.device
    )

    # 模型信息（必要）
    print(f"  Input  : {input_layer.any_name} "
          f"{input_layer.shape} {input_layer.element_type}")
    print(f"  Output : {output_layer.any_name} "
          f"{output_layer.shape} {output_layer.element_type}")

    # 步骤 3: LFW 验证
    run_lfw_test(
        compiled_model,
        input_layer,
        output_layer,
        args.lfw,
        args.num_people,
        args.num_images,
        args.threshold
    )


if __name__ == "__main__":
    main()
