# data_loader.py
# 负责数据集的加载和预处理

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_data(batch_size=1, n_samples_train=100, n_samples_test=50):
    """
    加载MNIST数据集，并进行预处理
    
    参数:
    batch_size: 批处理大小
    n_samples_train: 每个类别的训练样本数
    n_samples_test: 每个类别的测试样本数
    
    返回:
    train_loader: 训练数据加载器
    test_loader: 测试数据加载器
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 加载MNIST训练数据
    X_train = datasets.MNIST(
        root="./data", train=True, download=True, 
        transform=transforms.Compose([transforms.ToTensor()])
    )
    
    # 仅保留标签0和1
    idx = np.append(
        np.where(X_train.targets == 0)[0][:n_samples_train], 
        np.where(X_train.targets == 1)[0][:n_samples_train]
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    
    # 定义训练数据加载器
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    
    # 加载MNIST测试数据
    X_test = datasets.MNIST(
        root="./data", train=False, download=True, 
        transform=transforms.Compose([transforms.ToTensor()])
    )
    
    # 仅保留标签0和1
    idx = np.append(
        np.where(X_test.targets == 0)[0][:n_samples_test], 
        np.where(X_test.targets == 1)[0][:n_samples_test]
    )
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]
    
    # 定义测试数据加载器
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def visualize_samples(data_loader, n_samples=6):
    """
    可视化数据样本
    
    参数:
    data_loader: 数据加载器
    n_samples: 要显示的样本数量
    """
    data_iter = iter(data_loader)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(10, 3))
    
    samples_left = n_samples
    while samples_left > 0:
        images, targets = next(data_iter)
        
        axes[n_samples - samples_left].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
        axes[n_samples - samples_left].set_xticks([])
        axes[n_samples - samples_left].set_yticks([])
        axes[n_samples - samples_left].set_title("Labeled: {}".format(targets[0].item()))
        
        samples_left -= 1
    
    plt.show() 