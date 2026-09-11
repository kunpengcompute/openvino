# 安装指南

本文档介绍如何在鲲鹏 950 服务器上，基于 openEuler 24.03 LTS SP4 环境编译并安装 OpenVINO。

## 环境要求

### 表1硬件环境要求

| 项目 | 要求 |
| ---- | ---- |
| CPU | 鲲鹏950处理器 |
| 内存 | 24*64GB |

### 表2操作系统要求

| 项目 | 要求 |
| ---- | ---- |
| 操作系统 | openEuler 24.03 LTS SP4 |

### 表3软件要求

| 项目 | 要求 |
| ---- | ---- |
| 架构 | aarch64 |
| 编译器 | GCC 12.3.1 / Clang 19.1.7|
| Python | 3.10 或以上版本|
| 构建工具 | CMake 3.26 或以上版本|

## 安装步骤

### 1. 安装编译依赖

```bash
dnf install -y \
    git git-lfs gcc gcc-c++ make cmake ninja-build  \
    python3 python3-pip python3-devel wget curl unzip tar gzip xz \
    bzip2 patch which file findutils procps-ng numactl numactl-devel \
    libstdc++-devel glibc-devel openssl-devel
```

### 2. 获取 OpenVINO 源码

建议固定OpenVINO版本为2026.2.1

```bash
mkdir -p /workspace
cd /workspace

git clone https://github.com/openvinotoolkit/openvino.git

cd openvino

# 切换到固定版本
git checkout 2026.2.1

# 拉取全部子模块
git submodule update --init --recursive
```

### 3. 编译安装OpenVINO

```bash
cd /workspace/openvino/

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_PREFIX=/opt/openvino-gcc \
    -DENABLE_PYTHON=ON \
    -DENABLE_WHEEL=ON

cd build

cmake --build . --parallel $(nproc)

cmake --install .
```
若使用Clang编译，需要修改以下参数：

```bash
-DCMAKE_C_COMPILER=/usr/bin/clang \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
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


## Open Model Zoo 安装

Open Model Zoo（OMZ）是 OpenVINO 生态中的开源模型资源库，提供了丰富的预训练模型及配套工具，涵盖计算机视觉、自然语言处理等多种典型应用场景。用户可以通过安装 Open Model Zoo 提供的 omz_downloader 工具，根据模型名称下载所需的预训练模型及其相关文件。

下载模型后，还可以结合 omz_converter 等工具将部分模型转换为 OpenVINO 支持的 IR 格式（.xml 和 .bin），从而用于后续的模型推理和性能测试。Open Model Zoo 可用于快速获取示例模型，便于验证 OpenVINO 的模型转换、推理和部署流程。

1. 拉取 Open Model Zoo 代码仓：

    ```bash
    mkdir -p /workspace
    cd /workspace

    git clone https://github.com/openvinotoolkit/open_model_zoo.git
    ```

2. 安装工具：


    进入 accuracy_checker 目录：

    ```bash
    cd open_model_zoo/tools/accuracy_checker/
    ```

    注释 requirements-extra.in 中的：

    ```
    # DNA sequence matching
    # parasail>=1.2.4;platform_system!="Windows"
    # parasail~=1.2.4;platform_system=="Windows"
    ```

    安装依赖：

    ```
    python3 -m pip install -r requirements-core.in -r requirements-extra.in -r requirements-test.in
    ```

    安装工具命令：

    ```bash
    cd ../..
    python3 -m pip install -r tools/model_tools/requirements.in

    python3 -m pip install -e tools/model_tools
    ```

    添加环境变量：

    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```