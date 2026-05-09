#!/usr/bin/env python3
"""
车辆部件掩码可视化脚本
将掩码叠加到原始车辆图片上进行可视化展示
"""

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 科研配色方案 - 基于matplotlib tab10和Set2调色板
PART_COLORS = {
    'windows': '#1f77b4',      # 蓝色 (tab10)
    'lights': '#ff7f0e',       # 橙色 (tab10)
    'wheels': '#2ca02c',       # 绿色 (tab10)
    'hood': '#d62728',         # 红色 (tab10)
    'door': '#9467bd',         # 紫色 (tab10)
    'doors': '#9467bd',        # 紫色 (tab10)
    'roof': '#8c564b',         # 棕色 (tab10)
    'grille': '#e377c2',       # 粉色 (tab10)
    'bumper': '#7f7f7f',       # 灰色 (tab10)
    'mirror': '#bcbd22',       # 黄绿色 (tab10)
    'mirrors': '#bcbd22',      # 黄绿色 (tab10)
    'license_plate': '#17becf', # 青色 (tab10)
    'trunk': '#aec7e8',        # 浅蓝 (Set2)
    'fender': '#ffbb78',       # 浅橙 (Set2)
    'default': '#c7c7c7'       # 浅灰
}


def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def load_boxes_json(json_path):
    """
    加载boxes格式的JSON文件
    格式: [{"category": "windows", "mask": [[0,1,...],...]}, ...]
    """
    with open(json_path, 'r') as f:
        parts = json.load(f)
    return parts


def load_mask_json(json_path):
    """
    加载mask格式的JSON文件
    格式: [[0, 0, 1, ...], [0, 1, 1, ...], ...]
    返回单个掩码数组
    """
    with open(json_path, 'r') as f:
        mask = json.load(f)
    return np.array(mask)


def visualize_single_mask(img_path, mask_path, output_path=None, title=None):
    """
    可视化单张图片和对应的mask格式掩码
    """
    # 加载图片
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # 加载掩码
    mask = load_mask_json(mask_path)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img)
    axes[0].set_title('原始图片', fontsize=14)
    axes[0].axis('off')
    
    # 掩码
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('掩码', fontsize=14)
    axes[1].axis('off')
    
    # 叠加
    # 将掩码缩放到图片尺寸
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(img.size, Image.Resampling.NEAREST)
    mask_resized = np.array(mask_resized) / 255.0
    
    # 创建彩色叠加
    overlay = img_array.copy()
    overlay[mask_resized > 0.5] = [255, 0, 0]  # 红色标记
    
    # 混合
    blended = (img_array * 0.7 + overlay * 0.3).astype(np.uint8)
    axes[2].imshow(blended)
    axes[2].set_title('叠加效果', fontsize=14)
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_boxes_mask(img_path, boxes_path, output_path=None, title=None, alpha=0.5):
    """
    可视化单张图片和对应的boxes格式掩码（带类别标签）
    科研配色版：处理重叠区域，使用tab10配色
    """
    from PIL import ImageFilter
    
    # 加载图片
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # 加载部件标注
    parts = load_boxes_json(boxes_path)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 预处理所有掩码，检测重叠
    all_masks = []
    legend_elements = []
    
    # 首先缩放所有掩码到图片尺寸
    for part in parts:
        category = part.get('category', 'unknown')
        mask = part.get('mask', [])
        mask_np = np.array(mask)
        
        if mask_np.size == 0:
            continue
            
        color_hex = PART_COLORS.get(category, PART_COLORS['default'])
        color_rgb = np.array(hex_to_rgb(color_hex))
        
        # 缩放掩码到图片尺寸
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_resized = mask_pil.resize(img.size, Image.Resampling.NEAREST)
        mask_resized = (np.array(mask_resized) > 127).astype(np.float32)
        
        all_masks.append((mask_resized, color_rgb, color_hex, category))
        legend_elements.append(mpatches.Patch(color=color_hex, label=category))
    
    # 检测重叠区域
    overlap_mask = np.zeros(img_array.shape[:2], dtype=np.int32)
    for mask_resized, _, _, _ in all_masks:
        overlap_mask += (mask_resized > 0.5).astype(np.int32)
    
    overlap_area = overlap_mask > 1  # 重叠区域
    overlap_count = overlap_area.sum()
    
    # 创建叠加图层 - 每个像素只取第一个匹配的部件
    overlay = img_array.copy().astype(np.float32)
    assigned = np.zeros(img_array.shape[:2], dtype=bool)  # 已分配的像素
    
    # 按顺序分配，避免重叠
    for mask_resized, color_rgb, color_hex, category in all_masks:
        mask_area = (mask_resized > 0.5) & (~assigned)  # 只分配未分配的像素
        overlay[mask_area] = overlay[mask_area] * (1 - alpha) + color_rgb * alpha
        assigned |= mask_area
    
    # 绘制边界线
    overlay_uint8 = overlay.astype(np.uint8)
    overlay_img = Image.fromarray(overlay_uint8)
    
    for mask_resized, color_rgb, color_hex, category in all_masks:
        # 找到掩码边界
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        edges = mask_pil.filter(ImageFilter.FIND_EDGES)
        edges_np = np.array(edges) > 0
        
        # 边界使用深色
        edge_color = tuple(int(c * 0.6) for c in color_rgb)
        for y in range(edges_np.shape[0]):
            for x in range(edges_np.shape[1]):
                if edges_np[y, x]:
                    overlay_img.putpixel((x, y), edge_color)
    
    axes[1].imshow(overlay_img)
    title_str = f'Segmentation Mask ({len(parts)} parts)'
    if overlap_count > 0:
        title_str += f' [Overlap: {overlap_count}px]'
    axes[1].set_title(title_str, fontsize=14)
    axes[1].axis('off')
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9, 
                       framealpha=0.9, fancybox=True)
    
    # 纯掩码可视化（白底+彩色掩码，科研风格）
    mask_only = np.ones_like(img_array) * 255  # 白色背景
    assigned_mask = np.zeros(img_array.shape[:2], dtype=bool)
    
    for mask_resized, color_rgb, color_hex, category in all_masks:
        mask_area = (mask_resized > 0.5) & (~assigned_mask)
        mask_only[mask_area] = color_rgb
        assigned_mask |= mask_area
    
    axes[2].imshow(mask_only)
    axes[2].set_title('Mask Only', fontsize=14)
    axes[2].axis('off')
    if legend_elements:
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9,
                       framealpha=0.9, fancybox=True)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_parts(img_path, boxes_path, output_path=None, cols=4):
    """
    可视化每个部件的单独掩码 - 改进版
    """
    # 加载图片
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # 加载部件标注
    parts = load_boxes_json(boxes_path)
    n_parts = len(parts)
    
    if n_parts == 0:
        print("No parts found")
        return
    
    rows = (n_parts + cols) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, part in enumerate(parts):
        category = part.get('category', 'unknown')
        mask = part.get('mask', [])
        mask_np = np.array(mask)
        
        color_hex = PART_COLORS.get(category, PART_COLORS['default'])
        color_rgb = np.array(hex_to_rgb(color_hex))
        
        # 缩放掩码
        if mask_np.size > 0:
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_resized = mask_pil.resize(img.size, Image.Resampling.NEAREST)
            mask_resized = (np.array(mask_resized) > 127).astype(np.float32)
            
            # 创建叠加 - 更强的颜色
            overlay = img_array.copy().astype(np.float32)
            mask_area = mask_resized > 0.5
            overlay[mask_area] = overlay[mask_area] * 0.4 + color_rgb * 0.6
            
            # 绘制边界
            from PIL import ImageFilter
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            mask_pil2 = Image.fromarray(mask_uint8, mode='L')
            edges = mask_pil2.filter(ImageFilter.FIND_EDGES)
            edges_np = np.array(edges) > 0
            
            overlay_uint8 = overlay.astype(np.uint8)
            overlay_img = Image.fromarray(overlay_uint8)
            draw = ImageDraw.Draw(overlay_img)
            edge_color = tuple(int(c * 0.3) for c in color_rgb)
            for y in range(edges_np.shape[0]):
                for x in range(edges_np.shape[1]):
                    if edges_np[y, x]:
                        overlay_img.putpixel((x, y), edge_color)
            
            axes[idx].imshow(overlay_img)
        else:
            axes[idx].imshow(img)
        
        axes[idx].set_title(f'{category}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # 隐藏空白子图
    for idx in range(n_parts, len(axes)):
        axes[idx].axis('off')
    
    img_name = Path(img_path).stem
    fig.suptitle(f'{img_name} - All Parts', fontsize=16)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='车辆部件掩码可视化')
    parser.add_argument('--image', type=str, required=True, help='车辆图片路径')
    parser.add_argument('--mask', type=str, default=None, help='掩码文件路径 (mask格式)')
    parser.add_argument('--boxes', type=str, default=None, help='掩码文件路径 (boxes格式)')
    parser.add_argument('--output', type=str, default=None, help='输出图片路径')
    parser.add_argument('--mode', type=str, default='overview', 
                        choices=['overview', 'all_parts'],
                        help='可视化模式: overview=概览, all_parts=显示每个部件')
    parser.add_argument('--alpha', type=float, default=0.5, help='掩码透明度 (0-1)')
    
    args = parser.parse_args()
    
    img_path = Path(args.image)
    
    if not img_path.exists():
        print(f"图片不存在: {img_path}")
        return
    
    if args.boxes:
        boxes_path = Path(args.boxes)
        if not boxes_path.exists():
            print(f"掩码文件不存在: {boxes_path}")
            return
        
        if args.mode == 'all_parts':
            visualize_all_parts(img_path, boxes_path, args.output)
        else:
            visualize_boxes_mask(img_path, boxes_path, args.output, alpha=args.alpha)
    
    elif args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            print(f"掩码文件不存在: {mask_path}")
            return
        
        visualize_single_mask(img_path, mask_path, args.output)
    
    else:
        print("请指定 --mask 或 --boxes 参数")
        return


if __name__ == '__main__':
    main()
