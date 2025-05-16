# model_utils.py
# 负责模型保存和加载

import torch
from hybrid_model import HybridModel

def save_model(model, filepath="model.pt"):
    """
    保存模型状态
    
    参数:
    model: 要保存的模型
    filepath: 保存路径
    """
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存到 {filepath}")

def load_model(model_class=HybridModel, filepath="model.pt"):
    """
    加载模型状态
    
    参数:
    model_class: 模型类
    filepath: 模型文件路径
    
    返回:
    加载了权重的模型
    """
    # 创建一个新模型实例
    model = model_class()
    
    # 加载权重
    model.load_state_dict(torch.load(filepath))
    
    print(f"模型已从 {filepath} 加载")
    
    return model 