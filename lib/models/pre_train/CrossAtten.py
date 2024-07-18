import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3) 

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if attn.dim() == 4:
                #mask = mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
                mask = mask.unsqueeze(1).unsqueeze(1).expand_as(attn)
            attn.masked_fill_(mask, -float('inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_kv, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.normq = norm_layer(dim)
        self.normk = norm_layer(dim)
        self.normv = norm_layer(dim)

        self.attn = CrossAttention(dim, dim, num_kv, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv):
        xq = xq + self.drop_path(self.attn(self.normq(xq), self.normk(xk), self.normv(xv)))
        xq = xq + self.drop_path(self.mlp(self.norm2(xq)))
        return xq
    
class CoTransformer(nn.Module):
    def __init__(self, seqlen=16, num_joints=17, num_words=16 ,embed_dim=256,):
        super().__init__()
        depth = 3
        self.seqlen = seqlen
        self.num_joints = num_joints
        self.temp_pos_emb = nn.Parameter(torch.rand((1, seqlen, embed_dim)))
        self.spa_pos_emb = nn.Parameter(torch.rand((1, num_joints, embed_dim)))
        self.text_pos_emb = nn.Parameter(torch.rand((1, num_words, embed_dim)))

        self.block = nn.ModuleList([
            Block(dim=256, num_heads=8, mlp_hidden_dim=256*4., qkv_bias=True, 
                  qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)

    def get_attention_mask(self, caption_mask, joint_dim):
        B = caption_mask.shape[0]
        mask = torch.zeros((B, joint_dim), device=caption_mask.device) # [B, TJ]
        atten_mask = torch.cat([mask, caption_mask], dim=-1)           # [B, TJ+N]

        return atten_mask.bool()

    def forward(self, joint_feat, text_feat, caption_mask):
        """
        joint_feat      : [B, T, J, dim]
        text_feat       : [B, N, dim]
        caption_mask    : [B, 36]
        """
        B, T, J, C = joint_feat.shape
        
        # Pos embedding
        joint_feat = joint_feat + self.temp_pos_emb[:, :, None].tile(1, 1, self.num_joints, 1) \
            + self.spa_pos_emb[:, None, :].tile(1, self.seqlen, 1, 1)

        joint_feat = joint_feat.reshape(B, T*J, C)
        text_feat = text_feat + self.text_pos_emb

        atten_mask = self.get_attention_mask(caption_mask ,T*J)

        x = torch.cat([joint_feat, text_feat], dim=1)
        for blk in self.block:
            x = blk(x, atten_mask)

        x = self.norm(x)

        x = x[:, :T*J].reshape(B, T, J, -1)
        return x