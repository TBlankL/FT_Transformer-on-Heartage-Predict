"""
FT-Transformer训练示例
用于年龄预测的回归任务
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from ft_transformer import FTTransformer
from r2_loss import R2Loss, NegativeR2Loss, WeightedR2Loss

warnings.filterwarnings("ignore")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 结果保存目录
RESULT_DIR = '/work1/Ljt/FT-Transformer/FT_on_heartage/Result'



class HeartAgeDataset(Dataset):
    """心脏年龄数据集"""
    
    def __init__(self, categorical_data, numerical_data, targets, sample_weights=None):
        self.categorical_data = torch.LongTensor(categorical_data)
        self.numerical_data = torch.FloatTensor(numerical_data)
        self.targets = torch.FloatTensor(targets)
        self.sample_weights = torch.FloatTensor(sample_weights) if sample_weights is not None else None
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if self.sample_weights is not None:
            return (
                self.categorical_data[idx],
                self.numerical_data[idx],
                self.targets[idx],
                self.sample_weights[idx]
            )
        else:
            return (
                self.categorical_data[idx],
                self.numerical_data[idx],
                self.targets[idx]
            )

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    
    demo = pd.read_csv(r"/work1/Ljt/FT-Transformer/FT_on_heartage/Data/demographic2.csv")
    df = pd.read_csv(r"/work1/Ljt/FT-Transformer/FT_on_heartage/Data/new_panel_50w.bio.1017.csv")
    npnd = pd.read_csv(r"/work1/Ljt/FT-Transformer/FT_on_heartage/Data/no_protein_no_disease.csv")


    # 从demo中选择eid和age列，合并到df中
    demo_age = demo[['eid', 'age']].copy()
    # 如果df中已有age列，先删除它，使用demo中的age
    if 'age' in df.columns:
        df = df.drop(columns=['age'])
    df = df.merge(demo_age, on='eid', how='left')    
    df = df.merge(npnd, on='eid', how='inner')
    
    # 打印df的列名
    print(f"原始数据形状: {df.shape}")
    print(f"年龄范围: {df['age'].min()} - {df['age'].max()}")
    
    # 过滤小样本年龄
    min_samples = 10
    age_counts = df.groupby('age').size()
    small_sample_ages = age_counts[age_counts < min_samples].index
    df = df[~df['age'].isin(small_sample_ages)].copy()
    
    print(f"过滤后数据形状: {df.shape}")
    print(f"过滤后年龄范围: {df['age'].min()} - {df['age'].max()}")
    
    # 计算样本权重（用于加权R²损失）
    age_counts_filtered = df.groupby('age').size()
    max_count = age_counts_filtered.max()
    sample_weights = df['age'].map(lambda x: max_count / age_counts_filtered[x])
    df['sample_weight'] = sample_weights
    
    print(f"样本权重范围: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    

    # 分类特征
    DISCRETE_COLS = ['DM_GENDER','DM_ETH_W','DM_ETH_A', 'DM_ETH_B', 'DM_ETH_O', 'LS_ALC', 'LS_SMK', 'LS_PA', 'LS_DIET', 'LS_SLP', 'DH_ILL', 'DH_DIAB', 'DH_HYPT', 
    'MH_CHOL', 'MH_BP', 'MH_INSU', 'SDI_SCORE', 'MH_BPTREAT', 'MH_STATIN', 'FH_DIAB', 'FH_HBP', 'FH_HD', 'FH_STR']
    NUM_COLS = [c for c in df.columns if c not in DISCRETE_COLS and c not in ['eid', 'age', 'sample_weight','DM_AGE']]
    
    
    # 分离特征和目标
    X_categorical = df[DISCRETE_COLS].values
    X_numerical = df[NUM_COLS].values
    y = df['age'].values
    sample_weights = df['sample_weight'].values
    eids = df['eid'].values  # 保存eid
    
    # 数据已经预处理完成，直接返回
    return X_categorical, X_numerical, y, sample_weights, eids, DISCRETE_COLS, NUM_COLS

def create_model(categories, num_continuous, device):
    """创建FT-Transformer模型"""
    model = FTTransformer(
        categories=categories,
        num_continuous=num_continuous,
        dim=64,                    # 减少token embedding 维度 (64->32)
        depth=2,                   # 减少Transformer 层数 (2->1)
        heads=2,                   
        dim_head=8,            
        dim_out=1,                 
        attn_dropout=0.1,
        ff_dropout=0.1,
        num_residual_streams=1    
    ).to(device)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, use_weights=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if use_weights and len(batch) == 4:
            x_cat, x_cont, y, weights = batch
            weights = weights.to(device)
        else:
            x_cat, x_cont, y = batch[:3]
            weights = None
        
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        preds = model(x_cat, x_cont)
        
        if use_weights and weights is not None:
            loss = criterion(preds.squeeze(), y, weights)
        else:
            loss = criterion(preds.squeeze(), y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 每10个批次清理一次内存
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

def evaluate(model, data_loader, criterion, device, use_weights=False):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            if use_weights and len(batch) == 4:
                x_cat, x_cont, y, weights = batch
                weights = weights.to(device)
            else:
                x_cat, x_cont, y = batch[:3]
                weights = None
            
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)
            
            preds = model(x_cat, x_cont)
            
            if use_weights and weights is not None:
                loss = criterion(preds.squeeze(), y, weights)
            else:
                loss = criterion(preds.squeeze(), y)
            
            total_loss += loss.item()
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # 计算评估指标
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    try:
        pearson_corr = pearsonr(all_targets, all_preds)[0]
    except:
        pearson_corr = 0.0
    
    return {
        'loss': total_loss / len(data_loader),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson_corr
    }

def clear_old_checkpoints(result_dir):
    """清理旧的检查点文件"""
    checkpoint_file = os.path.join(result_dir, 'best_ft_transformer_model.pth')
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("✓ 已清理旧的检查点文件")
        except Exception as e:
            print(f"⚠ 清理检查点文件失败: {e}")

def main():
    """主训练函数"""
    print("开始FT-Transformer训练...")
    
    # 创建结果目录（如果不存在）
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"结果保存目录: {RESULT_DIR}")
    
    # 清理旧的检查点文件
    clear_old_checkpoints(RESULT_DIR)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置CUDA内存管理
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("✓ 已清理CUDA缓存")
    
    # 加载数据
    X_categorical, X_numerical, y, sample_weights, eids, discrete_cols, num_cols = load_and_preprocess_data()
    
    # 计算分类特征的类别数
    categories = [len(np.unique(X_categorical[:, i])) for i in range(len(discrete_cols))]
    num_continuous = len(num_cols)
    
    print(f"分类特征数量: {len(discrete_cols)}")
    print(f"连续特征数量: {num_continuous}")
    print(f"分类特征类别数: {categories}")
    
    # 划分数据集（同时保留eid）
    X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test, weights_train, weights_test, eids_train, eids_test = train_test_split(
        X_categorical, X_numerical, y, sample_weights, eids, test_size=0.2, random_state=42
    )
    
    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val, weights_train, weights_val, eids_train, eids_val = train_test_split(
        X_cat_train, X_num_train, y_train, weights_train, eids_train, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(y_train)}")
    print(f"验证集大小: {len(y_val)}")
    print(f"测试集大小: {len(y_test)}")
    
    # 创建数据加载器
    train_dataset = HeartAgeDataset(X_cat_train, X_num_train, y_train, weights_train)
    val_dataset = HeartAgeDataset(X_cat_val, X_num_val, y_val, weights_val)
    test_dataset = HeartAgeDataset(X_cat_test, X_num_test, y_test, weights_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)     
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   
    
    # 创建模型
    print("正在创建模型...")
    try:
        model = create_model(categories, num_continuous, device)
        print(f"✓ 模型创建成功")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return
    
    # 损失函数和优化器
    # 选择损失函数类型
    use_weights = True  # 使用样本权重
    criterion = WeightedR2Loss()
    print("使用加权R²损失函数")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # 训练循环
    epochs = 20
    best_val_r2 = float('-inf')  # R²越高越好，所以初始化为负无穷
    train_losses = []
    val_metrics_history = []
    
    print("\n开始训练...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} 开始...")
        
        # 训练
        try:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_weights)
            train_losses.append(train_loss)
            print(f"  训练完成，损失: {train_loss:.4f}")
        except Exception as e:
            print(f"  训练失败: {e}")
            break
        
        # 验证
        try:
            val_metrics = evaluate(model, val_loader, criterion, device, use_weights)
            val_metrics_history.append(val_metrics)
            print(f"  验证完成，R²: {val_metrics['r2']:.3f}")
        except Exception as e:
            print(f"  验证失败: {e}")
            break
        
        # 学习率调度
        scheduler.step()
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val R²: {val_metrics['r2']:.3f} | "
                  f"Val MAE: {val_metrics['mae']:.3f} | "
                  f"Val Pearson: {val_metrics['pearson']:.3f}")
        
        # 保存最佳模型
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            checkpoint_path = os.path.join(RESULT_DIR, 'best_ft_transformer_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': best_val_r2,
                'epoch': epoch,
                'categories': categories,
                'num_continuous': num_continuous,
                'discrete_cols': discrete_cols,
                'num_cols': num_cols
            }, checkpoint_path)
    
    print(f"\n训练完成! 最佳验证R²: {best_val_r2:.3f}")
    
    # 加载最佳模型进行测试
    checkpoint_path = os.path.join(RESULT_DIR, 'best_ft_transformer_model.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            # 检查特征数量是否匹配
            if (checkpoint.get('categories') == categories and 
                checkpoint.get('num_continuous') == num_continuous):
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ 成功加载最佳模型")
            else:
                print("⚠ 检查点特征数量不匹配，使用当前训练的模型")
        except Exception as e:
            print(f"⚠ 加载检查点失败: {e}，使用当前训练的模型")
    else:
        print("⚠ 未找到检查点文件，使用当前训练的模型")
    
    # 测试集评估
    test_metrics = evaluate(model, test_loader, criterion, device, use_weights)
    print(f"\n测试集结果:")
    print(f"MAE: {test_metrics['mae']:.3f}")
    print(f"RMSE: {test_metrics['rmse']:.3f}")
    print(f"R²: {test_metrics['r2']:.3f}")
    print(f"Pearson: {test_metrics['pearson']:.3f}")
    
    # 获取测试集预测结果用于可视化
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:  # 包含样本权重
                x_cat, x_cont, y, weights = batch
            else:  # 不包含样本权重
                x_cat, x_cont, y = batch
            
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            preds = model(x_cat, x_cont)
            test_preds.extend(preds.squeeze().cpu().numpy())
            test_targets.extend(y.numpy())
    
    # ========== 对所有数据进行预测 ==========
    print("\n开始对所有数据进行预测...")
    all_dataset = HeartAgeDataset(X_categorical, X_numerical, y, sample_weights)
    all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)
    
    model.eval()
    all_preds = []
    all_targets = []
    all_eids_list = []
    
    with torch.no_grad():
        sample_idx = 0  # 跟踪当前样本索引
        for batch in all_loader:
            if len(batch) == 4:  # 包含样本权重
                x_cat, x_cont, y_batch, weights = batch
            else:  # 不包含样本权重
                x_cat, x_cont, y_batch = batch
            
            batch_size = len(y_batch)
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            preds = model(x_cat, x_cont)
            
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
            # 获取当前batch对应的eid（使用样本索引）
            all_eids_list.extend(eids[sample_idx:sample_idx + batch_size])
            sample_idx += batch_size
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_eids = np.array(all_eids_list)
    
    print(f"✓ 完成对所有数据的预测，共 {len(all_preds)} 个样本")
    
    # 保存包含eid、真实值、预测值的CSV文件
    results_df = pd.DataFrame({
        'eid': all_eids,
        'y_true': all_targets,
        'y_pred': all_preds
    })
    csv_path = os.path.join(RESULT_DIR, 'all_predictions_with_eid.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ 预测结果已保存到: {csv_path}")

   # 保存结果 - 转换numpy类型为Python原生类型
    def convert_numpy_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    results = {
        'test_metrics': convert_numpy_types(test_metrics),
        'best_val_r2': float(best_val_r2),
        'categories': categories,
        'num_continuous': num_continuous,
        'discrete_cols': discrete_cols,
        'num_cols': num_cols
    }
    
    results_path = os.path.join(RESULT_DIR, 'ft_transformer_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 {results_path}")


if __name__ == "__main__":
    main()
