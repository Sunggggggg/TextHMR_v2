import os
import torch
import torch.nn as nn

from .operation.transformer import Transformer
from .operation.TIRG import TIRG
from .operation.GMM import GMM
from .operation.DSTformer import DSTformer
from .refine_regressor import R_Regressor

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
            
            # if pretrained_freeze :
            #     for p in model_backbone.parameters():
            #         p.requires_grad = False

        self.model_backbone = model_backbone

        # Fine-tuning stage
        self.proj_img = nn.Linear(2048, embed_dim)
        self.proj_joint = nn.Linear(num_joint*3, embed_dim)
        self.fusing = nn.Sequential(
            nn.LayerNorm(embed_dim*2), nn.Linear(embed_dim*2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.mask_transformer = GMM(seqlen, n_layers=n_layers, d_output=2048, d_model=embed_dim,
                                    num_head=8,  dropout=dropout, drop_path_r=drop_path_r, atten_drop=atten_drop, mask_ratio=0.5)
        # self.fusing = TIRG(input_dim=[embed_dim//2, embed_dim//2], output_dim=embed_dim//2)
        # self.hierical_transformer = Transformer(depth=3, embed_dim=embed_dim//2, mlp_ratio=4.,
        #     h=8, drop_rate=dropout, drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=self.stride)
        # self.r_regrossor = R_Regressor(embed_dim//2)

    def return_output(self, smpl_output, is_train):
        theta = smpl_output['theta']
        B = theta.shape[0]
        if not is_train :
            pred_cam = theta[..., :3].reshape(B, 3)
            pred_pose = theta[..., 3:75].reshape(B, 72)
            pred_shape = theta[..., 75:].reshape(B, 10)
            pred_verts = smpl_output['verts'].reshape(B, 6890, 3)
        else :
            pred_cam = theta[..., :3].reshape(B, -1, 3)
            pred_pose = theta[..., 3:75].reshape(B, -1, 72)
            pred_shape = theta[..., 75:].reshape(B, -1, 10)
            pred_verts = smpl_output['verts'].reshape(B, -1, 6890, 3)

        return pred_verts, pred_pose, pred_shape, pred_cam

    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        # Prepare
        B, T = f_img.shape[:2]
        
        # Lifting feat
        lift3d_pos = self.model_backbone(pose_2d)     # [B, T, J, 3] 

        # Init (Use SPIN backbone)
        img_feat = self.proj_img(f_img)
        pose_feat = self.proj_joint(lift3d_pos.flatten(-2) / 1000)
        feat = torch.cat([img_feat, pose_feat], dim=-1)
        feat = self.fusing(feat)

        smpl_output_global, mask_ids, mem, pred_global = self.mask_transformer(feat, is_train=is_train, J_regressor=J_regressor)
        smpl_output_global = self.return_output(smpl_output_global, is_train)

        return lift3d_pos, smpl_output_global, mask_ids
    
