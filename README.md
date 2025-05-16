# MNIST Qiskit

这个项目使用混合量子-经典神经网络模型对MNIST数据集中的数字0和1进行分类。该模型结合了经典卷积神经网络和量子神经网络，实现了高精度的分类。

## 项目结构

项目被拆分为以下几个模块：

- `data_loader.py`: 数据加载和预处理
- `quantum_nn.py`: 量子神经网络定义
- `hybrid_model.py`: 混合量子-经典模型定义
- `trainer.py`: 模型训练和评估
- `model_utils.py`: 模型保存和加载
- `main.py`: 主程序

## 安装依赖

```bash
pip install qiskit==1.3.1 qiskit-machine-learning==0.8.2 torch torchvision matplotlib numpy
```

## 使用方法

### 训练新模型

```bash
python main.py --mode train --epochs 10 --model-path model.pt
```

参数说明：
- `--mode`: 运行模式，可选值为 `train`（训练）或 `test`（测试）
- `--epochs`: 训练轮数
- `--model-path`: 模型保存/加载路径
- `--no-visualize`: 添加此参数不显示训练数据的可视化

### 加载并测试已有模型

```bash
python main.py --mode test --model-path model.pt
```

## 模块功能详解

### data_loader.py

负责加载MNIST数据集并进行预处理，包括：
- 只保留数字0和1
- 创建数据加载器
- 提供数据可视化功能

### quantum_nn.py

定义量子神经网络，包括：
- 创建特征映射和参数化电路
- 定义可观测量
- 构建EstimatorQNN

### hybrid_model.py

定义混合量子-经典模型，包括：
- 经典CNN部分
- 量子神经网络部分
- 模型和优化器的配置

### trainer.py

负责模型训练和评估，包括：
- 训练循环
- 模型评估
- 可视化训练结果和预测

### model_utils.py

提供模型保存和加载功能

### main.py

主程序，提供命令行接口，协调各个模块的工作
