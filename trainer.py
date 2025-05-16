# trainer.py
# 负责模型训练和评估

import torch
from torch import no_grad
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_loader, optimizer, loss_func, epochs=10):
    """
    训练模型
    
    参数:
    model: 要训练的模型
    train_loader: 训练数据加载器
    optimizer: 优化器
    loss_func: 损失函数
    epochs: 训练轮数
    
    返回:
    loss_list: 每轮的平均损失
    """
    # 存储损失历史
    loss_list = []
    
    # 设置模型为训练模式
    model.train()
    
    # 开始训练
    for epoch in range(epochs):
        total_loss = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 梯度初始化为零
            optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = loss_func(output, target)
            
            # 反向传播
            loss.backward()
            
            # 优化权重
            optimizer.step()
            
            # 存储损失
            total_loss.append(loss.item())
        
        # 计算当前轮的平均损失
        epoch_loss = sum(total_loss) / len(total_loss)
        loss_list.append(epoch_loss)
        
        # 打印进度
        print("训练进度 [{:.0f}%]\t损失: {:.4f}".format(
            100.0 * (epoch + 1) / epochs, epoch_loss))
    
    return loss_list

def evaluate_model(model, test_loader, loss_func, batch_size=1):
    """
    评估模型
    
    参数:
    model: 要评估的模型
    test_loader: 测试数据加载器
    loss_func: 损失函数
    batch_size: 批处理大小
    
    返回:
    loss: 平均损失
    accuracy: 准确率
    """
    # 设置模型为评估模式
    model.eval()
    
    # 在无梯度环境中进行评估
    with no_grad():
        correct = 0
        total_loss = []
        
        for batch_idx, (data, target) in enumerate(test_loader):
            # 前向传播
            output = model(data)
            
            # 确保输出形状正确
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)
            
            # 获取预测结果
            pred = output.argmax(dim=1, keepdim=True)
            
            # 计算正确预测的数量
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 计算损失
            loss = loss_func(output, target)
            total_loss.append(loss.item())
    
    # 计算平均损失和准确率
    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = 100.0 * correct / len(test_loader) / batch_size
    
    print("测试数据性能:\n\t损失: {:.4f}\n\t准确率: {:.1f}%".format(avg_loss, accuracy))
    
    return avg_loss, accuracy

def plot_training_results(loss_list):
    """
    绘制训练结果
    
    参数:
    loss_list: 训练损失历史
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list)
    plt.title("混合神经网络训练收敛曲线")
    plt.xlabel("训练轮次")
    plt.ylabel("负对数似然损失")
    plt.grid(True)
    plt.show()

def visualize_predictions(model, test_loader, n_samples=6):
    """
    可视化模型预测结果
    
    参数:
    model: 训练好的模型
    test_loader: 测试数据加载器
    n_samples: 要显示的样本数量
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建图形
    fig, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(12, 3))
    
    # 计数器
    count = 0
    
    # 无梯度环境
    with no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if count == n_samples:
                break
                
            # 获取预测
            output = model(data[0:1])
            
            # 确保输出形状正确
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)
            
            # 获取预测类别
            pred = output.argmax(dim=1, keepdim=True)
            
            # 显示图像
            axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            
            # 设置标题显示预测结果
            if pred.item() == target.item():
                title_color = 'green'
            else:
                title_color = 'red'
                
            axes[count].set_title(
                f"预测: {pred.item()}", 
                color=title_color
            )
            
            # 增加计数器
            count += 1
    
    plt.tight_layout()
    plt.show() 