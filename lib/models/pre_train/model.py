import torch
import torch.nn as nn
import torch.nn.functional as F

from .STFormer import STFormer
from .CrossAtten import CoTransformer
from .text_encoder import TEncoder

class Model(nn.Module):
    def __init__(self, num_total_motion) :
        super().__init__()
        self.mid_frame = 8
        self.num_words = 36
        self.seqlen = 64
        self.st_fromer = STFormer(num_frames=self.seqlen, num_joints=17, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2, norm_layer=None, pretrained=False)
        
        self.text_encoder = TEncoder(depth=3, embed_dim=256, mlp_hidden_dim=256*4.,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36)
        
        self.co_former = CoTransformer(seqlen=self.seqlen, num_joints=17, num_words=36 ,embed_dim=256)
        
        self.joint_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 3)
        )

        self.text_head = nn.ModuleList([nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Dropout()),
                                         nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout()),
                                         nn.Linear(32*17, num_total_motion)])

    def extraction_features(self, pose_2d, text_embeds, return_joint=False):
        """
        pose_2d         : [B, T, J, 2]
        text_embeds     : [7693]
        """
        # Stage 1
        joint_feat = self.st_fromer(pose_2d, return_joint=False)  # [B, T, J, dim] 
        pred_text = self.text_prediction(joint_feat)              # [B, num_total_motion]
        max_pred_text = torch.argmax(pred_text, dim=-1)           # [B]
        
        # Padding
        text_emb, caption_mask = [], []
        for idx in max_pred_text:
            motion_feat = torch.tensor(text_embeds[idx][0])                  # [N, 768]
            n = motion_feat.shape[0]
            # Padding
            motion_feat = torch.cat([motion_feat] + [torch.zeros_like(motion_feat[0:1]) for _ in range(self.num_words-n)], dim=0)
            mask = torch.ones((self.num_words), device=pose_2d.device)
            mask[:n] = 0.

            text_emb.append(motion_feat)
            caption_mask.append(mask)                # n

        text_emb = torch.stack(text_emb, dim=0).float().cuda()              # [B, N, 768]
        caption_mask = torch.stack(caption_mask, dim=0).bool().cuda()       # [B, N]

        #
        text_feat = self.text_encoder(text_emb, caption_mask)               # [B, N, dim]
        joint_feat = self.co_former(joint_feat, text_feat, caption_mask)    # [B, T, J, dim]             
        if return_joint :
            pred_kp_3d = self.joint_head(joint_feat)                        # [B, T, J, 3] 
            return pred_kp_3d
        else :
            return joint_feat

    def text_prediction(self, joint_feat):
        """ Text predicting via joint features
        joint_feat : [B, T, J, dim]
        """
        x = joint_feat.mean(dim=1)

        x = self.text_head[0](x)               # [B, J, d]
        x = self.text_head[1](x)               # [B, J, d]
        x = x.flatten(-2)                      # [B, J*d]
        x = self.text_head[2](x)               # [B, num_total_motion]

        return x

    def forward(self, pose_2d, text_emb, caption_mask):
        """
        pose_2d      : [B, T, J, 3] z=1
        text_emb     : [B, N, 768]
        caption_mask : [B, 36]
        """
        # Stage 1.
        joint_feat = self.st_fromer(pose_2d, return_joint=False)  # [B, T, J, dim] 
        pred_text = self.text_prediction(joint_feat)              # [B, num_total_motion]

        # Stage 2.
        text_feat = self.text_encoder(text_emb, caption_mask)     # [B, N, dim]
        joint_feat = self.co_former(joint_feat, text_feat, caption_mask)
        pred_kp_3d = self.joint_head(joint_feat)                  # [B, T, J, 3] 

        return pred_text, pred_kp_3d