import json
import os
import random
from model import objectives
from model.cross_transformer import CrossTransformer
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import logging
from timm.models.vision_transformer import Block
from utils.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
from .attr_loss import CEL_Sigmoid

# import sys
# sys.path.append("./model")
from .qFormer import build_qtransformer

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)

        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'part' in args.loss_names:
            self.qformer = build_qtransformer(args).half()
            self.query_embed = nn.Embedding(args.num_queries, args.hidden_dim)
            self.query_text = ['Car windows', 'Car lights', 'Car wheels', 'Car mirrors', 'Car roof','Car doors']
            self.img_conv = nn.Conv1d(in_channels=576, out_channels=1, kernel_size=1)
        
        
        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        
        if 'attr' in args.loss_names:
            self.attr_classifer = nn.ModuleList([nn.Linear(512, 1) for i in range(28)])
            sample_weight = torch.Tensor([0.2635, 0.3086, 0.2010, 0.5209, 0.1125, 0.2810, 0.0467, 0.0503, 0.0958, 0.1409, 0.0548, 0.0966, 0.0577, 0.1230, 0.0674, 0.0357, 0.0211, 0.0215, 0.0633, 0.0284, 0.1811, 0.0199, 0.1238, 0.0735, 0.0341, 0.0110, 0.0211, 0.0552])
            self.attr_loss = CEL_Sigmoid(sample_weight=sample_weight,attr_idx=28)
            self.bn = nn.BatchNorm1d(28)

        if 'mim' in args.loss_names:
            self.visual_decoder = CrossTransformer(512, 8, 96, 
                                depth = 4, context_dim=512)
            self.decoder_pred = nn.Linear(512, 16**2 * 3, bias=True) # decoder to patch
            scale = 512 ** -0.5 # 1/sqrt(512)
            self.visual_decoder_pos_embed = nn.Parameter(scale * torch.randn(577, 512))
            self.mask_token = nn.Parameter(torch.randn(1, 1, 512))
            # breakpoint()
            
            # self.cross_attn_v = nn.MultiheadAttention(self.embed_dim,
            #                                         self.embed_dim // 64,
            #                                         batch_first=True)
            # self.cross_modal_transformer_v = Transformer(width=self.embed_dim,
            #                                            layers=args.cmtv_depth,
            #                                            heads=self.embed_dim //
            #                                            64)
            # scale_v = self.cross_modal_transformer_v.width**-0.5

            # self.ln_post_v = LayerNorm(self.embed_dim)

            # proj_std = scale_v * ((2 * self.cross_modal_transformer_v.layers)**-0.5)
            # attn_std = scale_v
            # fc_std = (2 * self.cross_modal_transformer_v.width)**-0.5
            # for block in self.cross_modal_transformer_v.resblocks:
            #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # # init cross attn
            # nn.init.normal_(self.cross_attn_v.in_proj_weight, std=attn_std)
            # nn.init.normal_(self.cross_attn_v.out_proj.weight, std=proj_std)

            # self.mim_head = nn.Sequential(
            #     OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
            #                 ('gelu', QuickGELU()),
            #                 ('ln', LayerNorm(self.embed_dim)),
            #                 ('fc', nn.Linear(self.embed_dim, args.stride_size**2 * 3, bias=True))]))
            # # init mlm head
            # nn.init.normal_(self.mim_head.dense.weight, std=fc_std)
            # nn.init.normal_(self.mim_head.fc.weight, std=proj_std)

            # self.cross_attn_v = nn.MultiheadAttention(self.embed_dim,
            #                                         self.embed_dim // 64,
            #                                         batch_first=True)
            # self.ln_post_v = LayerNorm(self.embed_dim)
            # # init cross attn
            # nn.init.normal_(self.cross_attn_v.in_proj_weight, std=attn_std)
            # nn.init.normal_(self.cross_attn_v.out_proj.weight, std=proj_std)

            # # MAE decoder specifics
            # decoder_embed_dim = 512
            # norm_layer=LayerNorm
            # self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)

            # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 577, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            # self.decoder_blocks = nn.ModuleList([
            #     Block(decoder_embed_dim, 8, 4, qkv_bias=True, norm_layer=norm_layer)
            #     for i in range(8)])

            # self.decoder_norm = norm_layer(decoder_embed_dim)
            # self.decoder_pred = nn.Linear(decoder_embed_dim, 16**2 * 3, bias=True) # decoder to patch

            # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(576**.5), cls_token=True)
            # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            # torch.nn.init.normal_(self.mask_token, std=.02)


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]

        args_dict = vars(self.args)

        loss_weights = {}
        self.update_ratio = 0.1
        for t in self.current_task:
            if t+"_loss_weight" in args_dict.keys():
                loss_weights[t] = args_dict[t+"_loss_weight"]
            else:
                loss_weights[t] = 1.0

        self.loss_weights = loss_weights
        self.losses = None

        print(f'Training Model with {self.current_task} tasks')
        print('loss weights', self.loss_weights, 'updata_ratio', self.update_ratio)

    
    def update_losses(self, current_losses):
        if self.losses == None:
            self.losses = current_losses
            return

        for loss_name, loss in self.losses.items():
            self.losses[loss_name] = (loss + current_losses[loss_name]) / 2
    
    
    def cross_former(self, q, k, v):
        # breakpoint()
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def cross_former_v(self, q, k, v):
        x = self.cross_attn_v(
                self.ln_pre_i(q),
                self.ln_pre_t(k),
                self.ln_pre_t(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_v(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post_v(x)
        return x
    
    def filter_partFeature(self,part_feature,part_text):
        new_part_feature = []
        for t in part_text:
            ind = self.query_text.index(t)
            new_part_feature.append(part_feature[ind])

        return torch.stack(new_part_feature)

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()
        current_losses = {}
        # breakpoint()
        if self.args.augmentation:
            images = batch['images'].view(-1,3,self.args.img_size[0],self.args.img_size[1])
            caption_ids = batch['caption_ids'].view(-1,77)
        else:
            images = batch['images']
            caption_ids = batch['caption_ids']
        # breakpoint()
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            itc_loss = objectives.compute_itc(i_feats, t_feats, logit_scale)
            ret.update({'itc_loss':itc_loss * self.loss_weights["itc"], 'itc_loss_ori':itc_loss})
            current_losses.update({'itc_loss':itc_loss})
        
        if 'part' in self.current_task:
            part_loss = 0.
            for i in range(len(i_feats)):
                j = i // 4
                if self.args.augmentation:
                    box_masks, part_ids ,part_label,box_masks_part,part_token,i2t_list,part_text = batch["boxes"][j]
                else:
                    box_masks, part_ids ,part_label,box_masks_part,part_token,i2t_list,part_text = batch["boxes"][i]
                # part_ids = batch["part_ids"]
                # breakpoint()
                box_masks = torch.Tensor(box_masks).flatten(1).to("cuda")
                part_ids = torch.stack(part_ids).to("cuda")
                # breakpoint()
                part_label = torch.stack(part_label).to("cuda")
                part_token = torch.stack(part_token).to("cuda")
                box_masks_part = torch.Tensor(box_masks_part).flatten(1).to("cuda")
                i2t_list = i2t_list.to("cuda")

                # part_feats = self.base_model.encode_text(part_ids)
                # image_feats_m = image_feats[i,1:,:].unsqueeze(0) * box_masks.unsqueeze(2)
                # image_feats_m = image_feats_m.half()

                image_feats_part = image_feats[i,1:,:].unsqueeze(0) * box_masks_part.unsqueeze(2)
                image_feats_part = image_feats_part.float()

                part_t_feats = self.base_model.encode_text(part_token)
                part_t_feats = part_t_feats.mean(1)

                # breakpoint()
                part_feats = self.qformer(i_feats[i].half()+part_t_feats,self.query_embed.weight.half())
                new_part_feats = self.filter_partFeature(part_feats,part_text)
                # breakpoint()

                try:
                    part_loss += objectives.compute_part_itc(self.img_conv(image_feats_part.half()).squeeze(1).float(), new_part_feats.float(), logit_scale,i2t_list)
                except ValueError as e:
                    print(f"Error: {e}")
                    breakpoint()
            ret.update({'part_loss':(part_loss/len(i_feats)) * self.loss_weights["part"], 'part_loss_ori':part_loss/len(i_feats)})
            current_losses.update({'part_loss':part_loss/len(i_feats)})

        
        if 'sdm' in self.current_task:
            # ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
            sdm_loss = objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)
            ret.update({'sdm_loss':sdm_loss * self.loss_weights["sdm"], 'sdm_loss_ori':sdm_loss})
            current_losses.update({'sdm_loss':sdm_loss})

        if 'cmpm' in self.current_task:
            # ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

            cmpm_loss = objectives.compute_cmpm(i_feats, t_feats, batch['pids'])
            ret.update({'cmpm_loss':cmpm_loss * self.loss_weights["cmpm"], 'cmpm_loss_ori':cmpm_loss})
            current_losses.update({'cmpm_loss':cmpm_loss})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            # ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
            # breakpoint()
            id_loss = objectives.compute_id(image_logits, text_logits, batch['pids'].view(-1))
            ret.update({'id_loss':id_loss * self.loss_weights["id"], 'id_loss_ori':id_loss})
            current_losses.update({'id_loss':id_loss})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids'].view(-1)).float().mean()
            text_precision = (text_pred == batch['pids'].view(-1)).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids'].view(-1,77)
            # breakpoint()

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            # ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            mlm_loss = objectives.compute_mlm(scores, mlm_labels)
            ret.update({'mlm_loss':mlm_loss * self.loss_weights["mlm"], 'mlm_loss_ori':mlm_loss})
            current_losses.update({'mlm_loss':mlm_loss})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'attr' in self.current_task:
            ture_label = batch["attr_label"].view(-1,28).float()
            # breakpoint()
            logits = torch.cat([self.attr_classifer[i](i_feats.half()) for i in range(28)], dim=1)
            bn_logits = self.bn(logits).float()
            attr_celoss = self.attr_loss(bn_logits,ture_label)
            # breakpoint()
            ret.update({'attr_loss':attr_celoss * self.loss_weights["attr"], 'attr_loss_ori':attr_celoss})
            current_losses.update({'attr_loss':attr_celoss})


        if 'mim' in self.current_task:
            # mae loss

            mask = batch["mask"].view(-1,576)
            # breakpoint()
            # for m in batch["mask"]:
            #     l = len(m)
            #     index = random.randint(0, l-1)
            #     mask.append(list(m.values())[index])
            #     mask.append(list(m.values())[index])
            #     mask.append(list(m.values())[index])
            #     mask.append(list(m.values())[index])
            #     breakpoint()
            mask = torch.Tensor(mask).cuda()

            # x, mask, ids_restore = self.base_model.encode_image(images, mask_ratio=0.5)
            x, mask, ids_restore = self.base_model.encode_image(images,mask_ratio=self.args.mim_masked_rate, mask=mask.flatten(1))

            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            x = x + self.visual_decoder_pos_embed[:x.size(1)]

            # # x = self.cross_former_v(x.half(),text_feats,text_feats)
            # # x = self.mim_head(x)
            x = self.visual_decoder(x.half(), text_feats)
            x = self.decoder_pred(x).float()

            x = x[:,1:,:]

            # images = images[::4].repeat(4,1,1,1)
            mim_loss = self.mim_loss(images, x, mask)
            ret.update({'mim_loss':mim_loss * self.loss_weights["mim"], 'mim_loss_ori':mim_loss})
            current_losses.update({'mim_loss':mim_loss})

            # mae loss
            # x, mask, ids_restore = self.base_model.encode_image(images, mask_ratio=0.75)
            
            # mask = []
            # for m in batch["mask"]:
            #     l = len(m)
            #     index = random.randint(0, l-1)
            #     mask.append(list(m.values())[index])
            #     mask.append(list(m.values())[index])
            #     mask.append(list(m.values())[index])
            #     mask.append(list(m.values())[index])
                

            # mask = torch.Tensor(mask).cuda()
            # # breakpoint()
            # x, mask, ids_restore = self.base_model.encode_image(images, mask=mask.flatten(1))

            # pred = self.forward_decoder(x, ids_restore, text_feats)  # [N, L, p*p*3]
            # mim_loss = self.mim_loss(images, pred, mask)

            # ret.update({'mim_loss':mim_loss * self.loss_weights["mim"], 'mim_loss_ori':mim_loss})
            # current_losses.update({'mim_loss':mim_loss})

        # breakpoint()
        # self.update_losses(current_losses)

        return ret
    
    def forward_decoder(self, x, ids_restore,text_feats):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = x.half()
        # text_feats = text_feats.float()
        # breakpoint()
        x = self.cross_attn_v(
                self.ln_pre_i(x),
                self.ln_pre_t(text_feats),
                self.ln_pre_t(text_feats),
                need_weights=False)[0] + x
            
        x = self.ln_post_v(x)

        # x = x.half()
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    
    
    def mim_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        # breakpoint()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_mim(self,batch):
        images = batch['images'].view(-1,3,self.args.img_size[0],self.args.img_size[1])
        caption_ids = batch['caption_ids'].view(-1,77)
        _, text_feats = self.base_model(images, caption_ids)

        x, mask, ids_restore = self.base_model.encode_image(images, mask=batch['mask'].view(-1,24,24).flatten(1))
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.visual_decoder_pos_embed[:x.size(1)]

        x = self.visual_decoder(x.half(), text_feats)
        x = self.decoder_pred(x).float()

        x = x[:,1:,:]

        return x[0].unsqueeze(0), images[0].unsqueeze(0), mask


    def forward_mlm(self,batch):
        images = batch['images'].view(-1,3,self.args.img_size[0],self.args.img_size[1])
        caption_ids = batch['caption_ids'].view(-1,77)
        _, text_feats = self.base_model(images, caption_ids)

        image_feats, text_feats = self.base_model(images, caption_ids)

        mlm_ids = batch['mlm_ids'].view(-1,77)
            # breakpoint()

        mlm_feats = self.base_model.encode_text(mlm_ids)

        x = self.cross_former(mlm_feats, image_feats, image_feats)

        x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

        scores = x.float().reshape(-1, self.args.vocab_size)
        pred = scores.max(1)[1]


        return pred
    
def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model


"""

def info_nce_loss(query, positive_key, negative_keys, temperature=0.07):
    # query: 查询特征 (batch_size, feature_dim)
    # positive_key: 正样本特征 (batch_size, feature_dim)
    # negative_keys: 负样本特征 (batch_size, num_negatives, feature_dim)
    
    # 对正样本进行相似度计算
    positive_logit = torch.sum(query * positive_key, dim=-1, keepdim=True)  # (batch_size, 1)
    
    # 对负样本进行相似度计算
    negative_logits = torch.einsum('nc,nkc->nk', [query, negative_keys])  # (batch_size, num_negatives)
    
    # 将正样本和负样本的相似度组合起来 (batch_size, 1 + num_negatives)
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    
    # 使用温度参数
    logits /= temperature
    
    # 创建标签，正样本的标签是0
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(query.device)
    
    # 使用交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss

"""