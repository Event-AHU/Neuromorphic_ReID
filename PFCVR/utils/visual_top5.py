import os
import textwrap
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_results(text_list, img_paths, text_ids, img_ids, similarity_matrix, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 字体配置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 20

    # 尺寸参数
    TEXT_WIDTH_RATIO = 3.5  # 文本区域宽度比例（相当于2.5个图片宽度）
    IMG_SPACING_RATIO = 0.2  # 图片间距（图片宽度的1/5）
    TARGET_SIZE = (300, 300)
    BORDER_WIDTH = 8
    fontsize = 24

    for idx, (query_text, text_id) in enumerate(zip(text_list, text_ids)):
        top5_indices = similarity_matrix[idx][:5]

        # 创建画布和网格布局
        fig = plt.figure(figsize=(30, 8), dpi=150)  # 增大画布宽度
        gs = GridSpec(1, 6, width_ratios=[TEXT_WIDTH_RATIO, 1, 1, 1, 1, 1], 
                      wspace=IMG_SPACING_RATIO)

        # 文本区域（占用2.5个图片宽度）
        text_ax = fig.add_subplot(gs[0])
        wrapped_text = textwrap.fill(query_text, width=35)  # 增加每行字数
        text_ax.text(0.5, 0.5, wrapped_text,
                    ha='left', va='center',
                    fontsize=fontsize,
                    linespacing=1.8,
                    wrap=True)
        text_ax.axis('off')

        # 图片区域（调整后的布局）
        for i, img_idx in enumerate(top5_indices):
            img_path = img_paths[img_idx]
            img_id = img_ids[img_idx]

            # 图片处理
            img = Image.open(img_path).convert('RGB')
            resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # 添加边框
            border_color = '#00CC00' if img_id == text_id else '#FF3333'
            bordered_img = ImageOps.expand(resized_img, border=BORDER_WIDTH, fill=border_color)

            # 使用网格定位
            ax = fig.add_subplot(gs[i+1])  # gs[1]到gs[5]
            ax.imshow(bordered_img)
            ax.set_title(f"Rank {i+1}\nID: {img_id}", 
                        fontsize=fontsize,
                        pad=10)
            ax.axis('off')

        # 最终布局调整
        plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
        
        # 保存结果
        source_img_path = img_paths[np.where(np.array(text_ids) == text_id)[0][0]]
        base_name = os.path.splitext(os.path.basename(source_img_path))[0]
        plt.savefig(os.path.join(output_dir, f"{base_name}_result.png"),
                   bbox_inches='tight',
                   facecolor='white',
                   dpi=150)
        plt.close()
        # breakpoint()

# 使用示例
# visualize_results(
#     text_list=text_descriptions,       # 文本描述列表
#     img_paths=image_paths,             # 图片路径列表
#     text_ids=text_ids,                 # 文本ID列表
#     img_ids=image_ids,                 # 图片ID列表
#     similarity_matrix=similarity_rank, # 相似度排序矩阵
#     output_dir="./results"             # 输出目录
# )