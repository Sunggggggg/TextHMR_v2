import os
import torch
import torch.nn as nn
import pandas as pd

from .regressor import Regressor
from .transformer import Transformer
from lib.models.pre_train.model import Model as pre_trained_model

class Model(nn.Module):
    def __init__(self, 
                 num_total_motion,
                 text_archive,
                 pretrained) :
        super().__init__()
        self.text_archive = text_archive

        self.proj_img = nn.Linear(2048, 512)
        self.proj_joint = nn.Linear(3, 256)

        self.temp_encoder = Transformer(depth=3, embed_dim=512, mlp_hidden_dim=1024, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=64)
        self.pre_trained_model = pre_trained_model(num_total_motion)
        self.regressor = Regressor(512+256*17)

        if pretrained :
            pretrained_dict = torch.load(pretrained)['gen_state_dict']

            self.pre_trained_model.load_state_dict(pretrained_dict)
            print(f'=> loaded pretrained model from \'{pretrained}\'')
    
    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        """
        pose_2d     : [B, T, J, 3]
        img_feat    : [B, T, 2048]
        """
        B, T = f_img.shape[:2]
        pose_2d = pose_2d[..., :2]

        pose_3d = self.pre_trained_model.extraction_features(pose_2d, self.text_archive, return_joint=True)   # [B, T, J, 3]
        joint_feat = self.proj_joint(pose_3d)               # [B, T, J, 256]
        joint_feat = joint_feat.flatten(-2)                 # [B, T, 32*J]

        img_feat = self.proj_img(f_img)
        img_feat = self.temp_encoder(img_feat)
        f = torch.cat([joint_feat, img_feat], dim=-1)       # [B, T, 512+32*J]

        if is_train:
            f = f
        else :
            f = f[:, T//2 : T//2 + 1]

        smpl_output = self.regressor(f, is_train=is_train, J_regressor=J_regressor)

        scores = None
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 64
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, pose_3d
    
def get_model(num_total_motion, text_archive, pretrained):
    model = Model(num_total_motion, text_archive, pretrained)

    return model