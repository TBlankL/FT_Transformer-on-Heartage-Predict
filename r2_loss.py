"""
自定义R²损失函数
用于直接优化R²指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class R2Loss(nn.Module):
    """
    自定义R²损失函数
    R² = 1 - (SS_res / SS_tot)
    其中：
    SS_res = Σ(y_true - y_pred)²  (残差平方和)
    SS_tot = Σ(y_true - y_mean)²  (总平方和)
    
    损失函数 = 1 - R² = SS_res / SS_tot
    """
    
    def __init__(self, reduction='mean'):
        super(R2Loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        """
        计算R²损失
        
        Args:
            y_pred: 预测值 (batch_size,)
            y_true: 真实值 (batch_size,)
        
        Returns:
            loss: R²损失值
        """
        # 确保输入是一维张量
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        # 计算均值
        y_mean = torch.mean(y_true)
        
        # 计算残差平方和 (SS_res)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        # 计算总平方和 (SS_tot)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        
        # 避免除零错误
        if ss_tot == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # 计算R²损失 = 1 - R² = SS_res / SS_tot
        r2_loss = ss_res / ss_tot
        
        return r2_loss

class NegativeR2Loss(nn.Module):
    """
    负R²损失函数
    直接最大化R²，损失函数 = -R²
    """
    
    def __init__(self, reduction='mean'):
        super(NegativeR2Loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        """
        计算负R²损失
        
        Args:
            y_pred: 预测值 (batch_size,)
            y_true: 真实值 (batch_size,)
        
        Returns:
            loss: 负R²损失值
        """
        # 确保输入是一维张量
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        # 计算均值
        y_mean = torch.mean(y_true)
        
        # 计算残差平方和 (SS_res)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        # 计算总平方和 (SS_tot)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        
        # 避免除零错误
        if ss_tot == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # 计算R²
        r2 = 1 - (ss_res / ss_tot)
        
        # 返回负R²作为损失（因为我们要最大化R²）
        return -r2

class WeightedR2Loss(nn.Module):
    """
    加权R²损失函数
    考虑样本权重的R²损失
    """
    
    def __init__(self, reduction='mean'):
        super(WeightedR2Loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, y_true, weights=None):
        """
        计算加权R²损失
        
        Args:
            y_pred: 预测值 (batch_size,)
            y_true: 真实值 (batch_size,)
            weights: 样本权重 (batch_size,)
        
        Returns:
            loss: 加权R²损失值
        """
        # 确保输入是一维张量
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        if weights is None:
            weights = torch.ones_like(y_true)
        else:
            weights = weights.squeeze()
        
        # 计算加权均值
        y_mean = torch.sum(weights * y_true) / torch.sum(weights)
        
        # 计算加权残差平方和
        ss_res = torch.sum(weights * (y_true - y_pred) ** 2)
        
        # 计算加权总平方和
        ss_tot = torch.sum(weights * (y_true - y_mean) ** 2)
        
        # 避免除零错误
        if ss_tot == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # 计算加权R²损失
        r2_loss = ss_res / ss_tot
        
        return r2_loss

# 测试函数
def test_r2_loss():
    """测试R²损失函数"""
    # 创建测试数据
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.1, 2.1, 2.9, 4.1, 4.9])
    
    # 测试R2Loss
    r2_loss_fn = R2Loss()
    loss1 = r2_loss_fn(y_pred, y_true)
    print(f"R2Loss: {loss1.item():.4f}")
    
    # 测试NegativeR2Loss
    neg_r2_loss_fn = NegativeR2Loss()
    loss2 = neg_r2_loss_fn(y_pred, y_true)
    print(f"NegativeR2Loss: {loss2.item():.4f}")
    
    # 测试WeightedR2Loss
    weighted_r2_loss_fn = WeightedR2Loss()
    weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0])  # 给第3个样本更高权重
    loss3 = weighted_r2_loss_fn(y_pred, y_true, weights)
    print(f"WeightedR2Loss: {loss3.item():.4f}")

if __name__ == "__main__":
    test_r2_loss()
