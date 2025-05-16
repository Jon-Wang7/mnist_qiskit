# main.py
# 主程序，执行完整的训练和测试流程

import torch

from data_loader import load_mnist_data, visualize_samples
from hybrid_model import get_model_and_optimizer
from trainer import train_model, evaluate_model, plot_training_results, visualize_predictions
from model_utils import save_model, load_model

def train_new_model(epochs=10, visualize_data=True, save_path="model.pt"):
    """
    训练一个新模型
    
    参数:
    epochs: 训练轮数
    visualize_data: 是否可视化数据
    save_path: 模型保存路径
    
    返回:
    model: 训练好的模型
    loss_list: 训练损失历史
    """
    print("开始训练新模型...")
    
    # 加载数据
    train_loader, test_loader = load_mnist_data()
    
    # 可视化训练数据
    if visualize_data:
        print("可视化训练数据样本:")
        visualize_samples(train_loader)
    
    # 获取模型和优化器
    model, optimizer, loss_func = get_model_and_optimizer()
    
    # 训练模型
    loss_list = train_model(model, train_loader, optimizer, loss_func, epochs)
    
    # 保存模型
    save_model(model, save_path)
    
    # 评估模型
    evaluate_model(model, test_loader, loss_func)
    
    # 可视化预测结果
    print("可视化预测结果:")
    visualize_predictions(model, test_loader)
    
    # 绘制训练损失
    plot_training_results(loss_list)
    
    return model, loss_list

def load_and_test_model(model_path="model.pt"):
    """
    加载模型并在测试集上进行评估
    
    参数:
    model_path: 模型文件路径
    """
    print(f"加载模型 {model_path} 并进行测试...")
    
    # 加载数据
    _, test_loader = load_mnist_data()
    
    # 加载模型
    model = load_model(filepath=model_path)
    
    # 评估模型
    _, loss_func = get_model_and_optimizer()[0:3:2]  # 只获取模型和损失函数
    evaluate_model(model, test_loader, loss_func)
    
    # 可视化预测结果
    print("可视化预测结果:")
    visualize_predictions(model, test_loader)
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MNIST分类的混合量子-经典模型")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="运行模式: 'train' 训练新模型, 'test' 测试已有模型")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数 (默认: 10)")
    parser.add_argument("--model-path", type=str, default="model.pt",
                        help="模型文件路径 (默认: model.pt)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="不可视化训练数据")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    if args.mode == "train":
        train_new_model(
            epochs=args.epochs,
            visualize_data=not args.no_visualize,
            save_path=args.model_path
        )
    else:  # test
        load_and_test_model(args.model_path) 