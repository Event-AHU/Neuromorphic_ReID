#!/usr/bin/env python3
"""
T2I_VeRi_new 数据集可视化分析脚本
支持：
1. 数据集统计信息
2. 图像+文本描述可视化
3. 车辆部件掩码可视化
4. 批量采样展示
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# 颜色配置
PART_COLORS = {
    'windows': '#4169E1',      # 皇家蓝
    'lights': '#FFD700',       # 金色
    'wheels': '#32CD32',       # 酸橙绿
    'hood': '#FF6347',         # 番茄红
    'door': '#9370DB',         # 中紫色
    'roof': '#20B2AA',         # 浅海绿
    'grille': '#FF8C00',       # 深橙色
    'bumper': '#DC143C',       # 深红色
    'mirror': '#00CED1',       # 深青色
    'license_plate': '#FF69B4', # 热粉色
    'trunk': '#8B4513',        # 马鞍棕
    'fender': '#556B2F',       # 橄榄色
    'default': '#808080'       # 灰色
}

def load_dataset(data_root):
    """加载数据集"""
    data_root = Path(data_root)
    anno_file = data_root / "reid_with_mask_prompt_and_boxes_filepath_prefixid_idsplit_70_30.json"
    
    with open(anno_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    return annotations, data_root

def get_statistics(annotations):
    """计算数据集统计信息"""
    stats = {
        'total_samples': len(annotations),
        'train_count': sum(1 for a in annotations if a.get('split') == 'train'),
        'val_count': sum(1 for a in annotations if a.get('split') == 'val'),
        'unique_ids': len(set(a['id'] for a in annotations)),
        'captions_per_image': [],
        'caption_lengths': [],
    }
    
    for anno in annotations:
        captions = anno.get('captions', [])
        stats['captions_per_image'].append(len(captions))
        for cap in captions:
            stats['caption_lengths'].append(len(cap.split()))
    
    return stats

def print_statistics(stats):
    """打印统计信息"""
    print("=" * 50)
    print("T2I_VeRi_new 数据集统计信息")
    print("=" * 50)
    print(f"总样本数: {stats['total_samples']}")
    print(f"训练集样本: {stats['train_count']}")
    print(f"验证集样本: {stats['val_count']}")
    print(f"唯一车辆ID数: {stats['unique_ids']}")
    print(f"平均每张图片描述数: {np.mean(stats['captions_per_image']):.2f}")
    print(f"描述长度统计 (词数):")
    print(f"  - 最小: {min(stats['caption_lengths'])}")
    print(f"  - 最大: {max(stats['caption_lengths'])}")
    print(f"  - 平均: {np.mean(stats['caption_lengths']):.2f}")
    print("=" * 50)

def plot_statistics(stats, output_dir):
    """绘制统计图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 训练/验证集分布
    ax1 = axes[0]
    splits = ['Train', 'Val']
    counts = [stats['train_count'], stats['val_count']]
    colors = ['#4CAF50', '#2196F3']
    ax1.bar(splits, counts, color=colors)
    ax1.set_title('Train/Val Split Distribution')
    ax1.set_ylabel('Number of Samples')
    for i, v in enumerate(counts):
        ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    # 2. 描述长度分布
    ax2 = axes[1]
    ax2.hist(stats['caption_lengths'], bins=30, color='#9C27B0', edgecolor='white', alpha=0.7)
    ax2.set_title('Caption Length Distribution')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(stats['caption_lengths']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(stats["caption_lengths"]):.1f}')
    ax2.legend()
    
    # 3. 每张图片描述数量分布
    ax3 = axes[2]
    cap_counts = Counter(stats['captions_per_image'])
    x = sorted(cap_counts.keys())
    y = [cap_counts[i] for i in x]
    ax3.bar(x, y, color='#FF9800')
    ax3.set_title('Captions per Image')
    ax3.set_xlabel('Number of Captions')
    ax3.set_ylabel('Number of Images')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'dataset_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"统计图表已保存至: {output_path}")

def load_part_boxes(boxes_path, data_root):
    """加载部件标注"""
    boxes_file = data_root / boxes_path
    if not boxes_file.exists():
        return None
    
    with open(boxes_file, 'r') as f:
        parts = json.load(f)
    return parts

def visualize_sample(anno, data_root, output_dir, show_parts=True):
    """可视化单个样本"""
    fig = plt.figure(figsize=(16, 8))
    
    # 加载图像
    img_path = data_root / anno['file_path']
    if not img_path.exists():
        # 尝试其他路径
        alt_path = data_root.parent / 'vehicle_data' / img_path.name
        if alt_path.exists():
            img_path = alt_path
    
    img = Image.open(img_path).convert('RGB')
    
    # 创建子图
    if show_parts:
        gs = GridSpec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.5, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_parts = fig.add_subplot(gs[0, 1])
        ax_text = fig.add_subplot(gs[1, :])
    else:
        gs = GridSpec(1, 2, width_ratios=[1, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[0, 1])
    
    # 显示原图
    ax_img.imshow(img)
    ax_img.set_title(f"ID: {anno['id']} | Split: {anno.get('split', 'N/A')}", fontsize=14)
    ax_img.axis('off')
    
    # 显示部件标注
    if show_parts:
        parts = load_part_boxes(anno['boxes'], data_root)
        if parts:
            img_parts = img.copy()
            overlay = Image.new('RGBA', img_parts.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            legend_elements = []
            for part in parts:
                category = part.get('category', 'unknown')
                mask = part.get('mask', [])
                color = PART_COLORS.get(category, PART_COLORS['default'])
                
                # 将mask转换为RGBA
                mask_np = np.array(mask)
                if mask_np.size > 0:
                    # 获取mask边界
                    rows = np.any(mask_np, axis=1)
                    cols = np.any(mask_np, axis=0)
                    if rows.any() and cols.any():
                        # 缩放mask到图像尺寸
                        h, w = mask_np.shape
                        scale_h = img_parts.size[1] / h
                        scale_w = img_parts.size[0] / w
                        
                        # 创建彩色掩码
                        color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        mask_img = Image.new('RGBA', img_parts.size, (*color_rgb, 100))
                        mask_resized = Image.new('L', (w, h))
                        
                        # 绘制mask
                        for y in range(h):
                            for x in range(w):
                                if mask_np[y, x]:
                                    mask_resized.putpixel((x, y), 255)
                        
                        mask_resized = mask_resized.resize(img_parts.size, Image.Resampling.NEAREST)
                        overlay.paste(mask_img, (0, 0), mask_resized)
                        
                        legend_elements.append(mpatches.Patch(color=color, label=category))
            
            img_parts = Image.alpha_composite(img_parts.convert('RGBA'), overlay)
            ax_parts.imshow(img_parts)
            ax_parts.set_title('Part Annotations', fontsize=14)
            ax_parts.axis('off')
            
            if legend_elements:
                ax_parts.legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            ax_parts.text(0.5, 0.5, 'No part annotations', ha='center', va='center', fontsize=12)
            ax_parts.axis('off')
    
    # 显示文本描述
    ax_text.axis('off')
    captions = anno.get('captions', [])
    prompt = anno.get('prompt', '')
    
    text_content = f"Prompt: {prompt}\n\n"
    text_content += "Captions:\n"
    for i, cap in enumerate(captions, 1):
        text_content += f"{i}. {cap}\n"
    
    ax_text.text(0.02, 0.95, text_content, transform=ax_text.transAxes, fontsize=10,
                 verticalalignment='top', wrap=True, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_dir) / f"sample_{anno['id']}_{Path(anno['file_path']).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_batch(annotations, data_root, output_dir, n_samples=9, split=None):
    """批量可视化样本"""
    # 筛选
    if split:
        samples = [a for a in annotations if a.get('split') == split]
    else:
        samples = annotations
    
    # 随机采样
    samples = random.sample(samples, min(n_samples, len(samples)))
    
    n_cols = 3
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (ax, anno) in enumerate(zip(axes, samples)):
        img_path = data_root / anno['file_path']
        if not img_path.exists():
            alt_path = data_root.parent / 'vehicle_data' / img_path.name
            if alt_path.exists():
                img_path = alt_path
        
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        
        caption = anno.get('captions', [''])[0][:60] + '...'
        ax.set_title(f"ID: {anno['id']}\n{caption}", fontsize=9)
        ax.axis('off')
    
    # 隐藏空白子图
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'T2I_VeRi_new Samples ({split if split else "all"})', fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'batch_samples_{split if split else "all"}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"批量可视化已保存至: {output_path}")

def visualize_id_distribution(annotations, output_dir):
    """可视化ID分布（每个ID的样本数量）"""
    id_counts = Counter(a['id'] for a in annotations)
    counts = list(id_counts.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图
    ax1 = axes[0]
    ax1.hist(counts, bins=50, color='#3F51B5', edgecolor='white', alpha=0.7)
    ax1.set_title('Distribution of Samples per Vehicle ID')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Number of Vehicle IDs')
    ax1.axvline(np.mean(counts), color='red', linestyle='--', 
                label=f'Mean: {np.mean(counts):.1f}')
    ax1.legend()
    
    # Top 20 IDs
    ax2 = axes[1]
    top_ids = id_counts.most_common(20)
    ids = [str(x[0]) for x in top_ids]
    values = [x[1] for x in top_ids]
    ax2.barh(ids, values, color='#E91E63')
    ax2.set_title('Top 20 Vehicle IDs by Sample Count')
    ax2.set_xlabel('Number of Samples')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'id_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ID分布图已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='T2I_VeRi_new 数据集可视化分析')
    parser.add_argument('--data_root', type=str, 
                        default='/data/kongweizhe/VPT2I_V2/data/T2I_VeRi_new',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, 
                        default='./visual_t2iveri_new',
                        help='输出目录')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['stats', 'sample', 'batch', 'all'],
                        help='运行模式')
    parser.add_argument('--sample_id', type=int, default=None,
                        help='指定可视化的样本ID')
    parser.add_argument('--n_samples', type=int, default=9,
                        help='批量可视化时的样本数')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', None],
                        help='指定数据集划分')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    annotations, data_root = load_dataset(args.data_root)
    print(f"加载完成，共 {len(annotations)} 条记录")
    
    # 根据模式执行
    if args.mode in ['stats', 'all']:
        stats = get_statistics(annotations)
        print_statistics(stats)
        plot_statistics(stats, output_dir)
        visualize_id_distribution(annotations, output_dir)
    
    if args.mode in ['sample', 'all']:
        if args.sample_id is not None:
            # 查找指定ID的样本
            samples = [a for a in annotations if a['id'] == args.sample_id]
            if samples:
                for sample in samples[:3]:  # 最多显示3个
                    path = visualize_sample(sample, data_root, output_dir)
                    print(f"样本可视化已保存至: {path}")
            else:
                print(f"未找到ID为 {args.sample_id} 的样本")
        else:
            # 随机选择样本
            sample = random.choice(annotations)
            path = visualize_sample(sample, data_root, output_dir)
            print(f"样本可视化已保存至: {path}")
    
    if args.mode in ['batch', 'all']:
        visualize_batch(annotations, data_root, output_dir, 
                       n_samples=args.n_samples, split=args.split)
    
    print("\n可视化分析完成！")

if __name__ == '__main__':
    main()
