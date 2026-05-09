import sys
sys.path.append('..')
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import build_dataloader
from model import build_model
from utils.options import get_args
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize
from torchvision import transforms

# 初始化tokenizer
tokenizer = SimpleTokenizer()

# 定义特殊token的ID
sot_token = tokenizer.encoder["<|startoftext|>"]
eot_token = tokenizer.encoder["<|endoftext|>"]  # 修复：使用正确的结束标记
mask_token = tokenizer.encoder["<|mask|>"]


def decode_tokens(tokens):
    """将token ID转换为文本"""
    # 将tokens转换为numpy数组以便处理
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    
    # 过滤掉填充值(通常是0)并解码tokens
    tokens = [token for token in tokens if token != 0]
    text = tokenizer.decode(tokens)
    return text


def visualize_mlm_restoration(loader, model):
    """可视化MLM掩码文本的恢复结果"""
    model.eval()
    
    # 创建保存目录
    save_dir = "./mlm_visual"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
                
            # 将数据移到GPU
            batch = {k: (v.to('cuda') if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # 获取原始图片
            original_image = batch['images'][0].cpu()
            
            # 获取原始文本
            original_tokens = batch['caption_ids'][0]
            
            # 获取掩码文本
            mlm_tokens = batch['mlm_ids'][0]
            mlm_labels = batch['mlm_labels'][0]
            
            # 通过模型获取预测结果
            # 注意：这里需要根据您的模型结构调整
            predicted_tokens = model.forward_mlm(batch)

            result = torch.where(mlm_tokens == 49405, predicted_tokens[0:len(original_tokens)], mlm_tokens)
            
            # 解码文本
            original_text = decode_tokens(original_tokens)
            predicted_text = decode_tokens(result)
            masked_text = decode_tokens(mlm_tokens)
            
            with open(os.path.join(save_dir, batch["image_path"][0].split("/")[-1].split(".")[0]+".txt"),"w",encoding="utf-8") as f:
                f.writelines("ori_text:"+original_text+"\n"+"pre_text:"+predicted_text+"\n"+"masked_text:"+masked_text)
            


def prepare_model(args, num_classes, chkpt_dir):
    """准备模型"""
    model = build_model(args, num_classes)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    return model.cuda()


if __name__ == '__main__':
    args = get_args()
    args.MLM = True
    args.img_aug = True
    name = args.name
    
    # 构建数据加载器
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    
    # 准备模型
    model = prepare_model(args, num_classes, chkpt_dir="data/best.pth")
    
    # 可视化MLM恢复结果
    visualize_mlm_restoration(train_loader, model)  # 此处添加了调用注释，原指令可能需要添加冒号，但当前行语法完整，推测可能是注释中提及的冒号缺失，不过未发现上下文相关情况，代码保持原样。