# 安装指南

本文档介绍如何在鲲鹏950服务器上，基于openEuler 24.03 LTS SP4环境编译并安装OpenVINO。

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
| Python | 3.10或以上版本|
| 构建工具 | CMake 3.26或以上版本|

## 安装步骤

### 1. 安装编译依赖

```bash
dnf install -y \
    git git-lfs gcc gcc-c++ make cmake ninja-build  \
    python3 python3-pip python3-devel wget curl unzip tar gzip xz \
    bzip2 patch which file findutils procps-ng numactl numactl-devel \
    libstdc++-devel glibc-devel openssl-devel
```

### 2. 获取OpenVINO源码

建议固定OpenVINO版本为2026.2.1

创建工作目录：

```bash
mkdir -p /workspace
cd /workspace
```

拉取OpenVINO代码仓：

```bash
git clone https://github.com/openvinotoolkit/openvino.git
```

切换到固定版本:

```bash
cd openvino

git checkout 2026.2.1
```

拉取全部子模块:

```bash
git submodule update --init --recursive
```

### 3. 编译安装OpenVINO

使用`GCC 12.3.1`编译，系统默认已安装GCC，编译命令如下：

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

安装目录为`/opt/openvino-gcc`

若使用`Clang 19.1.7`编译，需先安装Clang：

```bash
# 使用llvm-toolset-19包名安装
sudo dnf install -y llvm-toolset-19

source /opt/openEuler/llvm-toolset-19/enable

# 验证clang安装
clang --version
```

使用Clang编译，编译命令如下：

```bash
cd /workspace/openvino/

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_INSTALL_PREFIX=/opt/openvino-clang \
    -DENABLE_PYTHON=ON \
    -DENABLE_WHEEL=ON \
    -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF

cd build

cmake --build . --parallel $(nproc)

cmake --install .
```

安装目录为`/opt/openvino-clang`

### 4. 配置Runtime环境（二选一）

gcc编译：

```bash
export OPENVINO_INSTALL_DIR=/opt/openvino-gcc

source /opt/openvino-gcc/setupvars.sh
```

clang编译：

```bash
export OPENVINO_INSTALL_DIR=/opt/openvino-clang

source /opt/openvino-clang/setupvars.sh
```

### 5. 验证安装

运行以下脚本验证OpenVINO Runtime：

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

成功输出如下信息，表示OpenVINO安装成功：

```text
Available devices:
['CPU']
```

## Open Model Zoo安装

OMZ（Open Model Zoo）是OpenVINO生态中的开源模型资源库，提供了丰富的预训练模型及配套工具，
涵盖计算机视觉、自然语言处理等多种典型应用场景。用户可使用其提供的omz_downloader工具，根据模型名称下载所需的预训练模型及其相关文件。

下载模型后，可结合omz_converter等工具将部分模型转换为OpenVINO支持的IR格式（.xml和.bin），
用于后续的模型推理和性能测试。用户可借助Open Model Zoo快速获取示例模型，便于验证OpenVINO的模型转换、推理和部署流程。

1. 拉取Open Model Zoo代码仓：

    ```bash
    mkdir -p /workspace
    cd /workspace

    git clone https://github.com/openvinotoolkit/open_model_zoo.git
    ```

2. 安装工具：

    进入accuracy_checker目录：

    ```bash
    cd open_model_zoo/tools/accuracy_checker/
    ```

    注释requirements-extra.in中的：

    ```bash
    # DNA sequence matching
    # parasail>=1.2.4;platform_system!="Windows"
    # parasail~=1.2.4;platform_system=="Windows"
    ```

    安装依赖：

    ```bash
    python3 -m pip install -r requirements-core.in -r requirements-extra.in -r requirements-test.in
    ```

    安装工具：

    ```bash
    cd ../..
    python3 -m pip install -r tools/model_tools/requirements.in
    python3 -m pip install -e tools/model_tools
    ```

    配置环境变量：

    ```bash
    export PATH="$(python3 -m site --user-base)/bin:$PATH"
    ```

    验证安装：

    ```bash
    which omz_downloader
    omz_downloader --help
    ```

    > ![](public_sys-resources/icon-note.gif) **说明**：
    > 如需环境变量永久生效，可将环境变量配置写入~/.bashrc：
    >
    > ```bash
    > echo 'export PATH="$(python3 -m site --user-base)/bin:$PATH"' >> ~/.bashrc
    > source ~/.bashrc
    > ```
