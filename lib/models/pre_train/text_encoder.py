import torch
import torch.nn as nn
from .transformer import Transformer

class TEncoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36):
        super().__init__()
        self.input_proj = nn.Linear(768, embed_dim)
        self.transformer = Transformer(depth=depth, length=length, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, h=h, 
                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
    
    def forward(self, text_embed, caption_mask):
        feature = self.input_proj(text_embed)
        feature = self.transformer(feature, caption_mask)         # [B, N, 128]

        return feature