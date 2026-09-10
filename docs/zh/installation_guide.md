# 安装指南

本文档介绍如何在鲲鹏 950 服务器上，基于 openEuler 24.03 LTS SP3 环境编译并安装 OpenVINO。

## 环境要求

### 表1硬件环境要求

| 项目 | 要求 |
| ---- | ---- |
| CPU | 鲲鹏950处理器 |

### 表2操作系统要求

| 项目 | 要求 |
| ---- | ---- |
| 操作系统 | openEuler 24.03 LTS SP3 |

### 表3软件要求

| 项目 | 要求 |
| ---- | ---- |
| 架构 | aarch64 |
| 编译器 | GCC 12.3.1|
| Python | 3.10 或以上版本|
| 构建工具 | CMake 3.26 或以上版本|

## 安装步骤

### 1. 安装编译依赖

```bash
dnf install -y \
    git git-lfs gcc gcc-c++ make cmake ninja-build \
    python3 python3-pip wget curl unzip tar gzip xz bzip2 \
    patch which file findutils procps-ng numactl numactl-devel \
    libstdc++-devel glibc-devel openssl-devel
```

### 2. 获取 OpenVINO 源码

建议固定OpenVINO版本为2026.3.1

```bash
cd /workspace

git clone https://github.com/openvinotoolkit/openvino.git

cd openvino

# 切换到固定版本
git checkout 2026.3.1

# 拉取全部子模块
git submodule update --init --recursive
```

### 3. 编译安装OpenVINO

```bash
cd /workspace/openvino/

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_PREFIX=/opt/openvino-gcc \
    -DENABLE_PYTHON=ON \
    -DENABLE_WHEEL=ON

    cmake --build . --parallel $(nproc)
    cmake --install .
```

安装目录默认为：

```text
/opt/openvino-gcc
```

### 4. 配置 Runtime 环境

```bash
export OPENVINO_INSTALL_DIR=/opt/openvino-gcc

source /opt/openvino-gcc/setupvars.sh
```

### 5. 验证安装

运行以下脚本验证 OpenVINO Runtime：

```bash
python3 - <<'PY'
import openvino as ov

print("OpenVINO version:")
print(ov.get_version())

core = ov.Core()

print("Available devices:")
print(core.available_devices)
PY
```

成功输出如下信息，表示 OpenVINO 安装成功：

```text
Available devices:
['CPU']
```
