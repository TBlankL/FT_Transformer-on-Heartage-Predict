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
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from ft_transformer import FTTransformer
from r2_loss import R2Loss, NegativeR2Loss, WeightedR2Loss

warnings.filterwarnings("ignore")

# 结果保存目录
RESULT_DIR = '/work1/Ljt/FT-Transformer/FT_on_heartage/Result'
MODEL_PATH = os.path.join(RESULT_DIR, 'best_ft_transformer_model.pth')


class HeartAgeDataset(Dataset):
    """心脏年龄数据集"""
    
    def __init__(self, categorical_data, numerical_data, targets):
        self.categorical_data = torch.LongTensor(categorical_data)
        self.numerical_data = torch.FloatTensor(numerical_data)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.categorical_data[idx],
            self.numerical_data[idx],
            self.targets[idx]
        )


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    categories = checkpoint['categories']
    num_continuous = checkpoint['num_continuous']
    
    model = FTTransformer(
        categories=categories,
        num_continuous=num_continuous,
        dim=64,
        depth=2,
        heads=2,
        dim_head=8,
        dim_out=1,
        attn_dropout=0.1,
        ff_dropout=0.1,
        num_residual_streams=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ 模型加载成功")
    print(f"  分类特征数量: {len(categories)}")
    print(f"  连续特征数量: {num_continuous}")
    print(f"  最佳验证R²: {checkpoint.get('val_r2', 'N/A')}")
    
    return model, checkpoint


def plot_prediction_scatter(y_true, y_pred, result_dir, save_path=None):
    """
    Plot prediction scatter plots (interface function)
    
    Parameters:
        y_true: Array of true ages
        y_pred: Array of predicted ages
        result_dir: Directory to save results
        save_path: Optional custom path to save the figure. If None, uses default name.
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # ========== First plot: real age (x) vs predict age (y) scatter plot with OLS fit ==========
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # Classify points relative to y = x
    above_mask = y_pred > y_true
    below_mask = y_pred < y_true
    equal_mask = np.isclose(y_pred, y_true)
    
    if np.any(above_mask):
        ax1.scatter(y_true[above_mask], y_pred[above_mask],
                    alpha=0.6, s=20, c='red', label='Residual > 0')
    if np.any(below_mask):
        ax1.scatter(y_true[below_mask], y_pred[below_mask],
                    alpha=0.6, s=20, c='blue', label='Residual < 0')
    if np.any(equal_mask):
        ax1.scatter(y_true[equal_mask], y_pred[equal_mask],
                    alpha=0.6, s=20, c='gray', label='Residual = 0')
    
    # OLS fitting
    ols_model = LinearRegression()
    ols_model.fit(y_true.reshape(-1, 1), y_pred)
    y_pred_ols = ols_model.predict(y_true.reshape(-1, 1))
    
    # Plot OLS fit line
    sorted_indices = np.argsort(y_true)
    ax1.plot(y_true[sorted_indices], y_pred_ols[sorted_indices], 
             'r-', linewidth=2, label=f'OLS fit (slope={ols_model.coef_[0]:.3f})')
    
    # Plot perfect prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'k--', linewidth=1.5, alpha=0.5, label='Perfect prediction (y=x)')
    
    # Calculate and display R²
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Real Age', fontsize=12)
    ax1.set_ylabel('Predict Age', fontsize=12)
    ax1.set_title('Real Age vs Predict Age (with OLS fit)', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    
    # Save first figure
    if save_path is None:
        save_path = os.path.join(result_dir, 'all_data_prediction_scatter_plots.png')
    fig1.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Scatter plot (ax1) saved to: {save_path}")
    plt.close(fig1)
    
    # ========== Second plot: predict age - OLS(real age) (y) vs real age (x) scatter plot ==========
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Calculate residuals: predict age - OLS(real age)
    residuals = y_pred - y_pred_ols
    
    # Classify by sign of residuals
    positive_mask = residuals > 0
    negative_mask = residuals < 0
    zero_mask = np.isclose(residuals, 0)
    
    # Plot scatter points: red for positive, blue for negative
    if np.any(positive_mask):
        ax2.scatter(y_true[positive_mask], residuals[positive_mask], 
                   alpha=0.6, s=20, c='red', label='Residual > 0')
    if np.any(negative_mask):
        ax2.scatter(y_true[negative_mask], residuals[negative_mask], 
                   alpha=0.6, s=20, c='blue', label='Residual < 0')
    if np.any(zero_mask):
        ax2.scatter(y_true[zero_mask], residuals[zero_mask], 
                   alpha=0.6, s=20, c='gray', label='Residual = 0')
    
    # Plot y=0 gray dashed line
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='y = 0')
    
    ax2.set_xlabel('Real Age', fontsize=12)
    ax2.set_ylabel('Predict Age - OLS(Real Age)', fontsize=12)
    ax2.set_title('Residuals: Predict Age - OLS(Real Age) vs Real Age', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    
    # Save second figure
    ax2_path = os.path.join(result_dir, 'all_data_prediction_residuals.png')
    fig2.savefig(ax2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Scatter plot (ax2) saved to: {ax2_path}")
    plt.close(fig2)


def main():
    """主函数：加载模型并对所有数据进行测试"""
    print("=" * 60)
    print("FT-Transformer 模型测试 - 所有数据")
    print("=" * 60)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return
    
    # 加载模型
    model, checkpoint = load_model(MODEL_PATH, device)
    
    # 读取数据
    print("\n正在加载数据...")
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
    
    # 使用checkpoint中保存的特征列名
    DISCRETE_COLS = checkpoint.get('discrete_cols', ['DM_GENDER','DM_ETH_W','DM_ETH_A', 'DM_ETH_B', 'DM_ETH_O', 'LS_ALC', 'LS_SMK', 'LS_PA', 'LS_DIET', 'LS_SLP', 'DH_ILL', 'DH_DIAB', 'DH_HYPT', 
    'MH_CHOL', 'MH_BP', 'MH_INSU', 'SDI_SCORE', 'MH_BPTREAT', 'MH_STATIN', 'FH_DIAB', 'FH_HBP', 'FH_HD', 'FH_STR'])
    NUM_COLS = checkpoint.get('num_cols', [c for c in df.columns if c not in DISCRETE_COLS and c not in ['eid', 'age', 'sample_weight','DM_AGE']])
    
    # 分离特征和目标
    X_categorical = df[DISCRETE_COLS].values
    X_numerical = df[NUM_COLS].values
    y = df['age'].values
    eids = df['eid'].values
    
    print(f"\n分类特征数量: {len(DISCRETE_COLS)}")
    print(f"连续特征数量: {len(NUM_COLS)}")
    print(f"总样本数: {len(y)}")
    
    # 创建数据集和数据加载器
    print("\n正在创建数据加载器...")
    all_dataset = HeartAgeDataset(X_categorical, X_numerical, y)
    all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)
    
    # 对所有数据进行预测
    print("\n开始对所有数据进行预测...")
    model.eval()
    all_preds = []
    all_targets = []
    all_eids_list = []
    
    with torch.no_grad():
        sample_idx = 0
        for batch_idx, batch in enumerate(all_loader):
            x_cat, x_cont, y_batch = batch
            
            batch_size = len(y_batch)
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            preds = model(x_cat, x_cont)
            
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_eids_list.extend(eids[sample_idx:sample_idx + batch_size])
            sample_idx += batch_size
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  已处理 {sample_idx}/{len(y)} 个样本...")
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_eids = np.array(all_eids_list)
    
    print(f"✓ 完成对所有数据的预测，共 {len(all_preds)} 个样本")
    
    # 计算评估指标
    print("\n计算评估指标...")
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    try:
        pearson_corr = pearsonr(all_targets, all_preds)[0]
    except:
        pearson_corr = 0.0
    
    print("\n" + "=" * 60)
    print("所有数据测试结果:")
    print("=" * 60)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"Pearson相关系数: {pearson_corr:.3f}")
    print("=" * 60)
    
    # 保存预测结果到CSV
    results_df = pd.DataFrame({
        'eid': all_eids,
        'y_true': all_targets,
        'y_pred': all_preds
    })
    csv_path = os.path.join(RESULT_DIR, 'all_data_predictions_with_eid.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✓ 预测结果已保存到: {csv_path}")
    
    # 绘制散点图
    print("\n正在绘制散点图...")
    plot_path = os.path.join(RESULT_DIR, 'all_data_prediction_scatter_plots.png')
    plot_prediction_scatter(all_targets, all_preds, RESULT_DIR, save_path=plot_path)
    
    # 保存评估指标
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson': float(pearson_corr),
        'num_samples': int(len(all_preds))
    }
    metrics_path = os.path.join(RESULT_DIR, 'all_data_test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ 评估指标已保存到: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()








