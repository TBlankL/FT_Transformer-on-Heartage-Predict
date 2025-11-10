"""
FT-Transformer 特征重要性分析
使用多种方法评估特征重要性：
1. 注意力权重分析（Attention-based importance）
2. 梯度重要性（Gradient-based importance）
3. 嵌入权重分析（Embedding weight analysis）
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from ft_transformer import FTTransformer
from FT_on_50wbio import HeartAgeDataset, load_and_preprocess_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 结果保存目录
RESULT_DIR = '/work1/Ljt/FT-Transformer/FT_on_heartage/Result'


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    categories = checkpoint['categories']
    num_continuous = checkpoint['num_continuous']
    
    model = FTTransformer(
        categories=categories,
        num_continuous=num_continuous,
        dim=64,
        depth=2,
        heads=4,
        dim_head=8,
        dim_out=1,
        attn_dropout=0.1,
        ff_dropout=0.1,
        num_residual_streams=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def attention_based_importance(model, data_loader, device, discrete_cols, num_cols):
    """
    基于注意力权重计算特征重要性
    使用CLS token对所有特征的注意力权重
    """
    print("正在计算基于注意力权重的特征重要性...")
    
    model.eval()
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 100:  # 只使用前100个批次以节省时间
                break
                
            x_cat, x_cont, y = batch[:3]
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            
            try:
                # 获取预测和注意力权重
                logits, attns = model(x_cat, x_cont, return_attn=True)
                
                # attns形状: [depth, batch, heads, seq_len, seq_len] 或 [batch, depth, heads, seq_len, seq_len]
                # 处理不同的维度顺序
                if attns.dim() == 5:
                    if attns.shape[0] == model.transformer.depth:  # [depth, batch, heads, seq_len, seq_len]
                        last_layer_attn = attns[-1]  # [batch, heads, seq_len, seq_len]
                    else:  # [batch, depth, heads, seq_len, seq_len]
                        last_layer_attn = attns[:, -1]  # [batch, heads, seq_len, seq_len]
                else:
                    last_layer_attn = attns  # 假设已经是最后一层
                
                # 平均所有头
                avg_attn = last_layer_attn.mean(dim=1)  # [batch, seq_len, seq_len]
                
                # CLS token (位置0) 对所有特征的注意力
                cls_attention = avg_attn[:, 0, 1:]  # [batch, num_features] (跳过CLS token自己)
                
                all_attention_weights.append(cls_attention.cpu().numpy())
            except Exception as e:
                print(f"警告: 批次 {batch_idx} 处理失败: {e}")
                continue
    
    if not all_attention_weights:
        print("警告: 无法计算注意力权重，返回零值")
        num_total = len(discrete_cols) + len(num_cols)
        zero_importance = np.zeros(num_total)
        return {
            'categorical': dict(zip(discrete_cols, zero_importance[:len(discrete_cols)])),
            'numerical': dict(zip(num_cols, zero_importance[len(discrete_cols):])),
            'all': dict(zip(discrete_cols + num_cols, zero_importance))
        }
    
    # 合并所有批次
    all_attention_weights = np.concatenate(all_attention_weights, axis=0)
    
    # 对所有样本取平均
    feature_importance = all_attention_weights.mean(axis=0)
    
    # 分离分类特征和连续特征
    num_cat_features = len(discrete_cols)
    num_num_features = len(num_cols)
    
    # 确保长度匹配
    if len(feature_importance) < num_cat_features + num_num_features:
        # 如果长度不匹配，进行填充或截断
        feature_importance = np.pad(feature_importance, (0, num_cat_features + num_num_features - len(feature_importance)))
    
    cat_importance = feature_importance[:num_cat_features]
    num_importance = feature_importance[num_cat_features:num_cat_features + num_num_features]
    
    return {
        'categorical': dict(zip(discrete_cols, cat_importance)),
        'numerical': dict(zip(num_cols, num_importance)),
        'all': dict(zip(discrete_cols + num_cols, feature_importance[:num_cat_features + num_num_features]))
    }


def gradient_based_importance(model, data_loader, device, discrete_cols, num_cols):
    """
    基于梯度计算特征重要性
    使用输入特征对输出的梯度
    """
    print("正在计算基于梯度的特征重要性...")
    
    model.eval()
    all_gradients_cat = []
    all_gradients_num = []
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 100:  # 只使用前100个批次以节省时间
            break
            
        x_cat, x_cont, y = batch[:3]
        x_cat = x_cat.to(device).requires_grad_(True)
        x_cont = x_cont.to(device).requires_grad_(True)
        
        # 前向传播
        logits = model(x_cat, x_cont)
        
        # 计算梯度
        if x_cat.requires_grad:
            grad_cat = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=x_cat,
                create_graph=False,
                retain_graph=True
            )[0]
            all_gradients_cat.append(grad_cat.abs().mean(dim=0).cpu().numpy())
        
        if x_cont.requires_grad:
            grad_num = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=x_cont,
                create_graph=False,
                retain_graph=False
            )[0]
            all_gradients_num.append(grad_num.abs().mean(dim=0).cpu().numpy())
    
    # 对所有批次取平均
    if all_gradients_cat:
        cat_grad_importance = np.mean(all_gradients_cat, axis=0)
    else:
        cat_grad_importance = np.zeros(len(discrete_cols))
    
    if all_gradients_num:
        num_grad_importance = np.mean(all_gradients_num, axis=0)
    else:
        num_grad_importance = np.zeros(len(num_cols))
    
    return {
        'categorical': dict(zip(discrete_cols, cat_grad_importance)),
        'numerical': dict(zip(num_cols, num_grad_importance)),
        'all': dict(zip(discrete_cols + num_cols, np.concatenate([cat_grad_importance, num_grad_importance])))
    }


def embedding_weight_importance(model, discrete_cols, num_cols):
    """
    基于嵌入层权重计算特征重要性
    分析嵌入权重的L2范数
    """
    print("正在计算基于嵌入权重的特征重要性...")
    
    importance = {}
    
    # 分类特征嵌入权重
    if hasattr(model, 'categorical_embeds') and model.categorical_embeds is not None:
        cat_embeddings = model.categorical_embeds.weight
        # 计算每个类别嵌入的L2范数
        cat_emb_norms = torch.norm(cat_embeddings, dim=1)
        
        # 对于每个分类特征，取该特征所有类别嵌入的平均L2范数
        # 需要根据categories_offset来确定每个特征对应的嵌入范围
        if hasattr(model, 'categories_offset'):
            offset = model.categories_offset.cpu().numpy()
            categories = model.num_unique_categories // len(discrete_cols) if len(discrete_cols) > 0 else 0
            
            cat_importance = []
            for i, col in enumerate(discrete_cols):
                if i < len(offset):
                    start_idx = int(offset[i])
                    # 估算每个特征的平均类别数（简化处理）
                    num_classes = 2  # 大多数分类特征是二元的
                    end_idx = min(start_idx + num_classes, len(cat_emb_norms))
                    if end_idx > start_idx:
                        feat_importance = cat_emb_norms[start_idx:end_idx].mean().item()
                    else:
                        feat_importance = cat_emb_norms[start_idx].item() if start_idx < len(cat_emb_norms) else 0.0
                    cat_importance.append(feat_importance)
                else:
                    cat_importance.append(0.0)
            
            importance['categorical'] = dict(zip(discrete_cols, cat_importance))
        else:
            importance['categorical'] = {col: 0.0 for col in discrete_cols}
    else:
        importance['categorical'] = {col: 0.0 for col in discrete_cols}
    
    # 连续特征嵌入权重
    if hasattr(model, 'numerical_embedder') and model.numerical_embedder is not None:
        num_emb_weights = model.numerical_embedder.weights  # [num_continuous, dim]
        num_emb_biases = model.numerical_embedder.biases  # [num_continuous, dim]
        
        # 计算每个连续特征嵌入的L2范数（权重和偏置的组合）
        num_emb_norms = torch.norm(num_emb_weights, dim=1) + torch.norm(num_emb_biases, dim=1) * 0.1
        num_importance = num_emb_norms.cpu().numpy()
        
        importance['numerical'] = dict(zip(num_cols, num_importance))
    else:
        importance['numerical'] = {col: 0.0 for col in num_cols}
    
    # 合并所有特征
    importance['all'] = {**importance['categorical'], **importance['numerical']}
    
    return importance


def plot_feature_importance(importance_dict, title, save_path, top_n=20):
    """绘制特征重要性图"""
    # 合并所有特征
    all_features = importance_dict['all']
    
    # 排序并取前top_n
    sorted_features = sorted(all_features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    feature_names, importance_values = zip(*sorted_features)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in importance_values]
    bars = plt.barh(range(len(feature_names)), importance_values, color=colors)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('重要性得分', fontsize=12)
    plt.title(f'{title} - Top {top_n} 特征', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表到: {save_path}")


def plot_comparison(attn_importance, grad_importance, emb_importance, save_path, top_n=20):
    """绘制三种方法的重要性对比图"""
    # 获取所有特征
    all_features = set(attn_importance['all'].keys())
    
    # 创建DataFrame
    comparison_data = []
    for feat in all_features:
        comparison_data.append({
            'feature': feat,
            'attention': attn_importance['all'].get(feat, 0),
            'gradient': grad_importance['all'].get(feat, 0),
            'embedding': emb_importance['all'].get(feat, 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 归一化每种方法的重要性（0-1范围）
    for col in ['attention', 'gradient', 'embedding']:
        max_val = df[col].abs().max()
        if max_val > 0:
            df[col + '_normalized'] = df[col] / max_val
        else:
            df[col + '_normalized'] = 0
    
    # 计算综合重要性（平均归一化值）
    df['combined'] = (df['attention_normalized'] + 
                     df['gradient_normalized'] + 
                     df['embedding_normalized']) / 3
    
    # 取前top_n
    df_top = df.nlargest(top_n, 'combined')
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 注意力重要性
    axes[0, 0].barh(range(len(df_top)), df_top['attention_normalized'], color='#FF6B6B')
    axes[0, 0].set_yticks(range(len(df_top)))
    axes[0, 0].set_yticklabels(df_top['feature'])
    axes[0, 0].set_xlabel('归一化重要性')
    axes[0, 0].set_title('注意力权重重要性', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. 梯度重要性
    axes[0, 1].barh(range(len(df_top)), df_top['gradient_normalized'], color='#4ECDC4')
    axes[0, 1].set_yticks(range(len(df_top)))
    axes[0, 1].set_yticklabels(df_top['feature'])
    axes[0, 1].set_xlabel('归一化重要性')
    axes[0, 1].set_title('梯度重要性', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. 嵌入权重重要性
    axes[1, 0].barh(range(len(df_top)), df_top['embedding_normalized'], color='#95E1D3')
    axes[1, 0].set_yticks(range(len(df_top)))
    axes[1, 0].set_yticklabels(df_top['feature'])
    axes[1, 0].set_xlabel('归一化重要性')
    axes[1, 0].set_title('嵌入权重重要性', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. 综合重要性
    axes[1, 1].barh(range(len(df_top)), df_top['combined'], color='#F38181')
    axes[1, 1].set_yticks(range(len(df_top)))
    axes[1, 1].set_yticklabels(df_top['feature'])
    axes[1, 1].set_xlabel('综合重要性')
    axes[1, 1].set_title('综合重要性（三种方法平均）', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图到: {save_path}")
    
    return df.sort_values('combined', ascending=False)


def main():
    """主函数"""
    print("=" * 60)
    print("FT-Transformer 特征重要性分析")
    print("=" * 60)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    checkpoint_path = os.path.join(RESULT_DIR, 'best_ft_transformer_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return
    
    print(f"\n正在加载模型: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)
    print("✓ 模型加载成功")
    
    # 获取特征名称
    discrete_cols = checkpoint['discrete_cols']
    num_cols = checkpoint['num_cols']
    
    print(f"\n分类特征数量: {len(discrete_cols)}")
    print(f"连续特征数量: {len(num_cols)}")
    
    # 加载数据（用于计算重要性）
    print("\n正在加载数据...")
    X_categorical, X_numerical, y, sample_weights, _, _ = load_and_preprocess_data()
    
    # 只使用一部分数据（前1000个样本）以加快计算
    sample_size = min(1000, len(y))
    X_categorical = X_categorical[:sample_size]
    X_numerical = X_numerical[:sample_size]
    y = y[:sample_size]
    
    dataset = HeartAgeDataset(X_categorical, X_numerical, y)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    print(f"使用 {sample_size} 个样本进行特征重要性分析")
    
    # 1. 注意力权重重要性
    print("\n" + "=" * 60)
    attn_importance = attention_based_importance(model, data_loader, device, discrete_cols, num_cols)
    print("✓ 注意力权重重要性计算完成")
    
    # 2. 梯度重要性
    print("\n" + "=" * 60)
    grad_importance = gradient_based_importance(model, data_loader, device, discrete_cols, num_cols)
    print("✓ 梯度重要性计算完成")
    
    # 3. 嵌入权重重要性
    print("\n" + "=" * 60)
    emb_importance = embedding_weight_importance(model, discrete_cols, num_cols)
    print("✓ 嵌入权重重要性计算完成")
    
    # 保存结果
    print("\n" + "=" * 60)
    print("正在保存结果...")
    
    results = {
        'attention_based': attn_importance,
        'gradient_based': grad_importance,
        'embedding_based': emb_importance
    }
    
    # 转换为可序列化格式
    def convert_to_serializable(d):
        if isinstance(d, dict):
            return {k: convert_to_serializable(v) for k, v in d.items()}
        elif isinstance(d, (np.ndarray, np.generic)):
            return float(d) if d.size == 1 else d.tolist()
        elif isinstance(d, (np.float32, np.float64)):
            return float(d)
        else:
            return d
    
    results_serializable = convert_to_serializable(results)
    
    results_path = os.path.join(RESULT_DIR, 'feature_importance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {results_path}")
    
    # 绘制图表
    print("\n正在生成可视化图表...")
    
    plot_feature_importance(
        attn_importance, 
        '注意力权重重要性',
        os.path.join(RESULT_DIR, 'feature_importance_attention.png'),
        top_n=30
    )
    
    plot_feature_importance(
        grad_importance,
        '梯度重要性',
        os.path.join(RESULT_DIR, 'feature_importance_gradient.png'),
        top_n=30
    )
    
    plot_feature_importance(
        emb_importance,
        '嵌入权重重要性',
        os.path.join(RESULT_DIR, 'feature_importance_embedding.png'),
        top_n=30
    )
    
    # 对比图
    comparison_df = plot_comparison(
        attn_importance,
        grad_importance,
        emb_importance,
        os.path.join(RESULT_DIR, 'feature_importance_comparison.png'),
        top_n=30
    )
    
    # 保存综合排名
    comparison_df.to_csv(
        os.path.join(RESULT_DIR, 'feature_importance_ranking.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    print("✓ 特征排名已保存到 CSV 文件")
    
    # 打印Top 20特征
    print("\n" + "=" * 60)
    print("Top 20 重要特征（综合排名）:")
    print("=" * 60)
    for idx, row in comparison_df.head(20).iterrows():
        print(f"{idx+1:2d}. {row['feature']:30s} | "
              f"综合: {row['combined']:.4f} | "
              f"注意力: {row['attention_normalized']:.4f} | "
              f"梯度: {row['gradient_normalized']:.4f} | "
              f"嵌入: {row['embedding_normalized']:.4f}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

