import os
import torch
import torch.nn as nn
from functools import partial

from .operation.transformer import Transformer
from .operation.GMM import GMM
from .operation.DSTformer import DSTformer

class Model(nn.Module):
    def __init__(self,
                 seqlen=64,
                 num_joint=17,
                 embed_dim=512,
                 n_layers=3,
                 dropout=0.1,
                 drop_path_r=0.2,
                 atten_drop=0.0,
                 pretrained='/mnt2/SKY/TextHMR_v2/pretrained_weight/best_epoch.bin'
                 ) :
        super().__init__()
        self.seqlen = seqlen
        self.stride = 9
        # Lifter load
        model_backbone = DSTformer()
        if pretrained:
            print('===> Loading checkpoint', pretrained)
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone.load_state_dict(checkpoint, strict=True)
            
            if True :
                for p in model_backbone.parameters():
                    p.requires_grad = False

        self.model_backbone = model_backbone

        # Fine-tuning stage
        self.proj_img = nn.Linear(2048, embed_dim)
        self.proj_joint = nn.Linear(embed_dim, embed_dim//4)
        self.proj_pose = nn.Linear(num_joint*embed_dim//4, embed_dim)
        self.fusing = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), 
            nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.mask_transformer = GMM(seqlen, n_layers=n_layers, d_output=2048, d_model=embed_dim,
                                    num_head=8, dropout=dropout, drop_path_r=drop_path_r, atten_drop=atten_drop, mask_ratio=0.25)      

    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        # Prepare
        B, T = f_img.shape[:2]
        
        # Lifting feat
        lift3d_pos = self.model_backbone.get_representation(pose_2d)           # [B, T, J, 3] 

        # Init
        img_feat = self.proj_img(f_img)
        pose_feat = self.proj_joint(lift3d_pos)             # [B, T, 17, 64]
        pose_feat = self.proj_pose(pose_feat.flatten(-2))   # [B, T, 512]
        feat = self.fusing(img_feat + pose_feat)

        smpl_output_global, mask_ids, mem, pred_global = self.mask_transformer(feat, is_train=is_train, J_regressor=J_regressor)

        if is_train:
            for s in smpl_output_global:
                s['theta'] = s['theta'].reshape(B, T, -1)
                s['verts'] = s['verts'].reshape(B, T, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, T, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, T, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, T, -1, 3, 3)
        else:
            for s in smpl_output_global:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        return None, smpl_output_global, mask_ids