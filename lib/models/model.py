import os
import torch
import torch.nn as nn

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
                 pretrained='/mnt/SKY/MotionBERT/checkpoint/pretrain/custom/best_epoch.bin'
                 ) :
        super().__init__()
        self.seqlen = seqlen
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
        self.proj_joint = nn.Linear(num_joint*3, embed_dim//2)
        self.proj_img_local = nn.Linear(2048, embed_dim//2)
        self.fusing = TIRG(input_dim=[embed_dim//2, embed_dim//2], output_dim=embed_dim//2)
        self.mask_transformer = GMM(seqlen, n_layers=n_layers, d_output=2048, d_model=2048,
                                    num_head=8,  dropout=dropout,  drop_path_r=drop_path_r, atten_drop=atten_drop, mask_ratio=0.5)
        self.r_regrossor = R_Regressor(embed_dim//2)

    def return_output(self, smpl_output):
        theta = smpl_output['theta']
        pred_cam, pred_pose, pred_shape = theta[..., :3], theta[..., 3:75], theta[..., 75:]
        pred_verts = smpl_output['verts']
        kp_3d = smpl_output['kp_3d']

        return pred_verts, pred_pose, pred_shape, pred_cam, kp_3d
        # return {'pred_verts':pred_verts, 'pred_pose':pred_pose, 'pred_shape':pred_shape,
        #          'pred_cam':pred_cam, 'kp_3d':kp_3d}

    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        B, T = f_img.shape[:2]
        # Lifting feat
        lift3d_pos = self.model_backbone(pose_2d)     # [B, T, J, 3] # mm
        lift3d_pos = lift3d_pos / 1000                # m

        # Init 
        img_feat = self.proj_img(f_img)
        smpl_output_global, mask_ids, mem, pred_global = self.mask_transformer(img_feat, is_train=is_train, J_regressor=J_regressor)
        smpl_output_global = self.return_output(smpl_output_global)

        # Fusing
        img_feat = self.proj_img_local(f_img)
        pose_feat = self.proj_joint(lift3d_pos.flatten(-2))              # [B, T, 512]
        feat = self.fusing(img_feat, pose_feat)                          # [B, T, 256]


        if is_train :
            feat = feat[:, :]
        else :
            feat

        smpl_output = self.r_regrossor(feat, pred_global[0], pred_global[1], pred_global[2], n_iter=3, is_train=is_train, J_regressor=J_regressor)
        smpl_output = self.return_output(smpl_output)

        return lift3d_pos, smpl_output_global, smpl_output, mask_ids