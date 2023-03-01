from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Residual Block
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        
        return x

# Attention 
class MultiHeadAttention(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.emb_size = args_dict['emb_size']
        self.num_heads = args_dict['num_heads']
        self.qkv = nn.Linear(self.emb_size, self.emb_size * 3)
        self.att_drop = nn.Dropout(args_dict['drop_p'])
        self.projection = nn.Linear(self.emb_size, self.emb_size)
        
    def forward(self, x, mask=None):

        # q = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.num_heads)
        # k = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.num_heads)
        # v = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
     
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav ', att, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# MLP
class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion, drop_p=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p),
        )
    def forward(self, x):
        return self.net(x)

# Patch Embedding
class PatchEmbeddings(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.patch_size = args_dict['patch_size']
        self.in_channels = args_dict['in_channels']
        self.emb_size = args_dict['emb_size']

        # position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.emb_size, kernel_size=self.patch_size, stride=self.patch_size), #  1,768,14,14
            Rearrange('b e (h) (w) -> b (h w) e'), # 1, 196, 768
        )
    
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.proj(x)

        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        
        return x

### LayerNorm -> MultiHead -> LayerNorm -> MLP ->  
###     |                  ^      |             ^
###     --------------------      ---------------

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, args_dict):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args_dict['emb_size']),
                MultiHeadAttention(args_dict),
                nn.Dropout(args_dict['drop_p'])
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args_dict['emb_size']),
                FeedForward(args_dict['emb_size'], args_dict['expansion'], args_dict['drop_p']),
                nn.Dropout(args_dict['drop_p'])
            ))
        )

class TransformerEncoder(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.TransformerEncoder = nn.Sequential(
            *[TransformerEncoderBlock(args_dict) for _ in range(args_dict['depth'])]
        )

    def forward(self, x):
        return self.TransformerEncoder(x)
    
class ClassificationHead(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.ClassificationHead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(args_dict['emb_size']),
            nn.Linear(args_dict['emb_size'], args_dict['n_classes'])
        )
    
    def forward(self, x):
        return self.ClassificationHead(x)

class Vit(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.Vit = nn.Sequential(
            PatchEmbeddings(args_dict),
            TransformerEncoder(args_dict),
            ClassificationHead(args_dict),
        )

    def forward(self, x):
        return self.Vit(x)