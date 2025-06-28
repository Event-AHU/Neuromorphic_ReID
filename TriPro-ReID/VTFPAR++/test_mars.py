import os
import torch
import torchvision.transforms as T
from PIL import Image
from CLIP.clip import clip
from models.base_block import TransformerClassifier
import argparse
import numpy as np
from tqdm import tqdm
import random

# 设置随机种子和设备
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(605)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device, download_root='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/CLIP-ReID-master/VTFPAR++')

# 属性名称列表
attr_words = [
    'top short', 'bottom short', 'shoulder bag', 'backpack',
    'hat', 'hand bag', 'long hair', 'female', 'bottom skirt',
    'frontal', 'lateral-frontal', 'lateral', 'lateral-back', 'back', 'pose varies',
    'walking', 'running', 'riding', 'staying', 'motion varies',
    'top black', 'top purple', 'top green', 'top blue', 'top gray', 'top white', 'top yellow', 'top red', 'top complex',
    'bottom white', 'bottom purple', 'bottom black', 'bottom green', 'bottom gray',
    'bottom pink', 'bottom yellow', 'bottom blue', 'bottom brown', 'bottom complex',
    'young', 'teenager', 'adult', 'old'
]


def argument_parser():
    parser = argparse.ArgumentParser(description='Image or Video Classification for CLIP')
    parser.add_argument('--img_path', type=str, default='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/mars/rgb/bbox_test/0006/0006C1T0001F002.jpg', help='Path to the image')
    parser.add_argument('--checkpoint_path', type=str, default='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/CLIP-ReID-master/VTFPAR++/VTF++_mars.pth', help='Path to the checkpoint')
    parser.add_argument('--length', type=int, default=224, help='Length for Transformer model')
    return parser


parser = argument_parser()
args = parser.parse_args()


normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    normalize
])


def ordered_random_sample(lst, N):
    indices = sorted(random.sample(range(len(lst)), N))
    return [lst[i] for i in indices]

ViT_model = ViT_model.to(device)
model = TransformerClassifier(ViT_model, attr_num=43, attr_words=attr_words).to(device)

checkpoint_path = args.checkpoint_path
if not os.path.exists(checkpoint_path):
    raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

id_folder = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/iLIDS-VID/rgb/sequences/cam1'

model.eval()
pred_attrs = {}
for ids in tqdm(os.listdir(id_folder)):
    ids_path = os.path.join(id_folder, ids)
    samples = ordered_random_sample(os.listdir(ids_path), 6)
    imgs = []
    for sample in samples:
        img = Image.open(os.path.join(id_folder, ids, sample))
        imgs.append(transform(img))
    img_tensor = torch.stack(imgs).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(img_tensor, ViT_model=ViT_model)
        pred_result = torch.sigmoid(logits[0]).cpu().numpy() > 0.45
        # labels = []
        # for i, v in enumerate(pred_result):
        #     if v:
        #         labels.append(attr_words[i])
        #     elif i==7:
        #         labels.append('male')
        pred_attrs[ids] = str(list(pred_result))
json_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/CLIP-ReID-master/ilids_attrlabels.json'
import json
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(pred_attrs, f, indent=2, ensure_ascii=False)

