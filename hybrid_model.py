# hybrid_model.py
# 定义混合量子-经典神经网络模型

import torch
from torch import cat
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
)
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector

from quantum_nn import create_qnn

class HybridModel(Module):
    """
    混合量子-经典神经网络模型
    
    结合了经典CNN和量子神经网络
    """
    def __init__(self):
        """
        初始化混合模型
        """
        super().__init__()
        
        # 创建量子神经网络并连接到PyTorch
        qnn = create_qnn()
        self.qnn = TorchConnector(qnn)  # 量子-经典接口
        
        # 定义卷积层
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        
        # 定义全连接层
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)  # 2维输入到量子神经网络
        self.fc3 = Linear(1, 1)  # 量子神经网络的1维输出

    def forward(self, x):
        """
        前向传播
        
        参数:
        x: 输入张量 [batch_size, 1, 28, 28]
        
        返回:
        输出张量 [batch_size, 2]
        """
        # 经典CNN部分
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # 量子神经网络部分
        x = self.qnn(x)
        
        # 输出层
        x = self.fc3(x)
        
        # 创建两类输出概率
        return cat((x, 1 - x), -1)


def get_model_and_optimizer():
    """
    创建模型和优化器
    
    返回:
    model: 混合模型
    optimizer: 优化器
    loss_func: 损失函数
    """
    # 创建模型
    model = HybridModel()
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = NLLLoss()
    
    return model, optimizer, loss_func 