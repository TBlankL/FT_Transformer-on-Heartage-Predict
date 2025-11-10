# R²损失函数使用指南

## 概述

本文档详细说明了如何在FT-Transformer训练中使用R²作为损失函数，以及不同R²损失函数变体的设定方法。

## R²损失函数类型

### 1. 标准R²损失函数 (R2Loss)

**数学公式：**
```
R² = 1 - (SS_res / SS_tot)
损失 = SS_res / SS_tot = 1 - R²
```

**特点：**
- 直接优化R²指标
- 损失值越小，R²越高
- 适合大多数回归任务

**使用设定：**
```python
loss_type = "r2"
criterion = R2Loss()
```

### 2. 负R²损失函数 (NegativeR2Loss)

**数学公式：**
```
损失 = -R²
```

**特点：**
- 直接最大化R²
- 损失值越小，R²越高
- 梯度方向更直观

**使用设定：**
```python
loss_type = "negative_r2"
criterion = NegativeR2Loss()
```

### 3. 加权R²损失函数 (WeightedR2Loss)

**数学公式：**
```
加权R² = 1 - (加权SS_res / 加权SS_tot)
损失 = 加权SS_res / 加权SS_tot
```

**特点：**
- 考虑样本权重
- 适合处理不平衡数据
- 对少数样本给予更高权重

**使用设定：**
```python
loss_type = "weighted_r2"
criterion = WeightedR2Loss()
```

## 配置方法

### 1. 修改损失函数类型

在 `train_ft_transformer.py` 的第350行修改：

```python
# 选择损失函数类型
loss_type = "r2"  # 可选: "mse", "r2", "negative_r2", "weighted_r2"
```

### 2. 损失函数对比

| 损失函数 | 优点 | 缺点 | 适用场景 |
|---------|------|------|----------|
| MSE | 简单稳定，梯度平滑 | 不直接优化R² | 一般回归任务 |
| R² | 直接优化R²指标 | 可能不稳定 | 关注R²的回归任务 |
| 负R² | 梯度方向直观 | 可能不稳定 | 直接最大化R² |
| 加权R² | 处理不平衡数据 | 需要计算权重 | 不平衡数据集 |

## 使用建议

### 1. 选择策略

**推荐顺序：**
1. **MSE损失** - 如果训练稳定且R²已经较好
2. **R²损失** - 如果希望直接优化R²指标
3. **负R²损失** - 如果R²损失不稳定
4. **加权R²损失** - 如果数据不平衡且其他方法效果不佳

### 2. 超参数调整

使用R²损失函数时，建议调整以下超参数：

```python
# 学习率可能需要调整
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)  # 降低学习率

# 增加训练轮数
epochs = 100  # 从50增加到100

# 使用更稳定的学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
```

### 3. 监控指标

使用R²损失函数时，重点关注：

- **训练损失** - 应该稳定下降
- **验证R²** - 主要优化目标
- **验证MAE** - 确保预测精度
- **梯度范数** - 避免梯度爆炸

## 实际使用示例

### 示例1：标准R²损失

```python
# 在train_ft_transformer.py中设置
loss_type = "r2"
use_weights = False

# 运行训练
python train_ft_transformer.py
```

### 示例2：加权R²损失（处理不平衡数据）

```python
# 在train_ft_transformer.py中设置
loss_type = "weighted_r2"
use_weights = True

# 样本权重会自动计算
# 权重 = max_count / age_count
```

### 示例3：负R²损失（直接最大化R²）

```python
# 在train_ft_transformer.py中设置
loss_type = "negative_r2"
use_weights = False

# 使用更小的学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

## 注意事项

### 1. 数值稳定性

- R²损失函数可能在某些情况下不稳定
- 建议使用梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### 2. 批次大小

- 小批次可能导致R²计算不稳定
- 建议使用较大的批次大小（64或更大）

### 3. 学习率

- R²损失函数通常需要较小的学习率
- 建议从1e-4开始尝试

### 4. 早停策略

```python
# 基于R²的早停
if val_metrics['r2'] > best_val_r2:
    best_val_r2 = val_metrics['r2']
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 20:  # 20个epoch无改善则停止
        break
```

## 结果分析

### 1. 损失曲线

- **MSE损失**：应该单调下降
- **R²损失**：应该单调下降（R²上升）
- **负R²损失**：应该单调下降（R²上升）

### 2. 性能对比

比较不同损失函数的最终结果：

```python
# 测试不同损失函数
loss_functions = ["mse", "r2", "negative_r2", "weighted_r2"]
results = {}

for loss_type in loss_functions:
    # 训练模型
    # 记录测试集R²
    results[loss_type] = test_r2
```

## 总结

R²损失函数提供了直接优化R²指标的方法，特别适合年龄预测等回归任务。选择合适的R²损失函数变体，结合适当的超参数调整，可以显著提升模型的R²性能。建议从标准R²损失开始尝试，根据训练稳定性调整到其他变体。
