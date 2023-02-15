from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.emb_size = args_dict['att_emb']
        self.num_heads = args_dict['num_heads']
        self.qkv = nn.Linear(self.emb_size, self.emb_size)
        self.att_drop = nn.Dropout(args_dict['drop_p'])
        self.projection = nn.Linear(self.emb_size, self.emb_size)
        
    def forward(self, x, mask=None):
        # print(x.shape) # (1, 197, 16) * (197, 197 * 3)
        # x, linear
        # (1, 197, 16) * (16, 16) = (1, 197, 16)
        x = rearrange(x, 'b n e -> b e n')
        print(self.qkv(x).shape)
        q = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.num_heads)
        # qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=1)
        # queries, keys, values = qkv[0], qkv[1], qkv[2]
     
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

class PatchEmbeddings(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.patch_size = args_dict['patch_size']
        self.img_size = args_dict['img_size']
        self.projection = nn.Sequential(
            nn.Conv2d(args_dict['in_channels'], self.patch_size, kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.cls_token = nn.Parameter(torch.randn(1,1, self.patch_size))
        self.pos = nn.Parameter(torch.randn((self.img_size // self.patch_size) ** 2 + 1, self.patch_size))

    def forward(self, x):
        b, C, H, W = x.shape
        x = self.projection(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos
        # x = rearrange(x, 'b n e -> b e n')
        print('first')
        print(x.shape)
        return x

### LayerNorm -> MultiHead -> LayerNorm -> MLP ->  
###     |                  ^      |             ^
###     --------------------      ---------------

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, args_dict):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm([args_dict['emb_size'], args_dict['patch_size']]),
                MultiHeadAttention(args_dict),
                nn.Dropout(args_dict['drop_p'])
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm([args_dict['emb_size'], args_dict['patch_size']]), 
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
            nn.LayerNorm([args_dict['emb_size'], args_dict['patch_size']]),
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
