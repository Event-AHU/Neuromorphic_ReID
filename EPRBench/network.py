
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class SGVPRNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, pretrained_foundation = False, 
                 foundation_model_path = None,
                 use_clip_text: bool = True,
                 clip_model_name: str = 'ViT-B-16',
                 clip_pretrained: str = 'laion2b_s34b_b79k',
                 clip_freeze: bool = True,
                 ):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())

        # In TransformerEncoderLayer, "batch_first=False" means the input tensors should be provided as (seq, batch, feature) to encode on the "seq" dimension.
        # Our input tensor is provided as (batch, seq, feature), which performs encoding on the "batch" dimension.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, 
            nhead=16,
            dim_feedforward=2048,
            activation="gelu", 
            dropout=0.1,
            batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # Cross-image encoder
        self.text_alpha = nn.Parameter(torch.tensor(0.5))
        self.text_alpha_patch = nn.Parameter(torch.tensor(0.5))        
        self.text_token_ln = nn.LayerNorm(768, eps=1e-6)
        self.mask_keep_ratio = 0.25
        if use_clip_text:
            from backbone.text_encoder_clip import CLIPTextEncoder
            self.text_encoder_clip = CLIPTextEncoder(
                model_name='ViT-B-16',
                pretrained='laion2b_s34b_b88k',
                output_dim=768,
                freeze_clip=True
            )
        else:
            self.text_encoder_clip = None
        
    def forward(self, x, text: torch.Tensor = None, text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None, raw_texts: list = None, return_aux: bool = False):
        x = self.backbone(x)        

        B,P,D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P - 1))
        x0 = x["x_norm_clstoken"]
        x_patch = x["x_prenorm"][:, 1:, :]  # 提取 patch tokens (B, N, D)

        image_global_raw = x0
        
        if self.text_encoder_clip is not None and raw_texts is not None:
            text = self.text_encoder_clip(raw_texts)  # (B, 768)
            tokens_out = self.text_encoder_clip.encode_batch_tokens(raw_texts)
            text_tokens = tokens_out['tokens'].to(x0.device)  # (B, L, D_clip)
            text_mask = tokens_out['mask'].to(x0.device)   # (B, L)
        
        # Patch Fusion
        if text_tokens is not None:
            device = x_patch.device
            B, N, D = x_patch.shape
            
            t_tokens = self.text_token_ln(text_tokens)
            x_patch_norm = F.normalize(x_patch, p=2, dim=-1)
            t_tokens_norm = F.normalize(t_tokens, p=2, dim=-1)

            sim_matrix = torch.bmm(x_patch_norm, t_tokens_norm.transpose(1, 2))

            if text_mask is not None:
                sim_matrix = sim_matrix.masked_fill(text_mask.unsqueeze(1) == 0, -1e9)

            max_sim, max_token_idx = sim_matrix.max(dim=-1)

            k = max(int(self.mask_keep_ratio * N), 1)

            _, topk_idx = torch.topk(max_sim, k=k, dim=1, largest=True, sorted=False)

            selected_x = torch.gather(x_patch, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
            selected_token_idx = torch.gather(max_token_idx, 1, topk_idx)
            selected_text_feats = torch.gather(t_tokens, 1, selected_token_idx.unsqueeze(-1).expand(-1, -1, D))
            enhanced_part = selected_x * selected_text_feats 
            x_out = x_patch.clone()
            x_out.scatter_add_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), self.text_alpha_patch * enhanced_part)
            
            x_patch = x_out
        
        x_p = x_patch.view(B,W,H,D).permute(0, 3, 1, 2) 

        x10,x11,x12,x13 = self.aggregation(x_p[:,:,0:8,0:8]),self.aggregation(x_p[:,:,0:8,8:]),self.aggregation(x_p[:,:,8:,0:8]),self.aggregation(x_p[:,:,8:,8:])
        x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.aggregation(x_p[:,:,0:5,0:5]),self.aggregation(x_p[:,:,0:5,5:11]),self.aggregation(x_p[:,:,0:5,11:]),\
                                        self.aggregation(x_p[:,:,5:11,0:5]),self.aggregation(x_p[:,:,5:11,5:11]),self.aggregation(x_p[:,:,5:11,11:]),\
                                        self.aggregation(x_p[:,:,11:,0:5]),self.aggregation(x_p[:,:,11:,5:11]),self.aggregation(x_p[:,:,11:,11:])
        x = [i.unsqueeze(1) for i in [x0,x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]

        x = torch.cat(x,dim=1)
        x = self.encoder(x).view(B,14*D)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        if return_aux:
            return x, image_global_raw
        return x

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone

