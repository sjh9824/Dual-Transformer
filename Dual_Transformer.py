"""This file is a combination of Physformer.py and transformer_layer.py
   in the official PhysFormer implementation here:
   https://github.com/ZitongYu/PhysFormer

   model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import numpy as np
from typing import Optional
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import math

"""
        Test =      'Basic': q,k,v = [B, num, T, C] -> T Attention
                    'Test1' : q k v = [B, num, C, T] -> C Attention
                    'Test2' : T Attention + C Attention
"""            
        
Test = 'Test2'
print_flag = False

class ChannelGroupAttention(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.qkv = nn.Sequential(nn.Linear(dim, dim , bias=False))
        self.bn = nn.Sequential(nn.BatchNorm1d(dim))
        
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim)
        )
        
        self.dim  = dim
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B,P,C] = x.shape #[B ,640, 96]
        print_flags(x, "CGA x.shape",print_flag,2)
        # x = x.transpose(1,2).view(B, C, P//16, 4, 4) #[B, 96, 40, 4, 4]
        # print_flags(x, "CGA After transpose",print_flag,2)
        q = self.qkv(x).transpose(1,2)
        k = self.qkv(x).transpose(1,2)
        v = self.qkv(x).transpose(1,2)
        print_flags(q, "CGA qkv",print_flag,2)
        
        
        q = self.bn(q)
        k = self.bn(k)
        v = self.bn(v)
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2) #[B, 96, 640]
        print_flags(q, "CGA flatten",print_flag,2)
        q, k ,v = (split_last(x, (self.n_heads, -1)).transpose(1,2) for x in [q,k,v])
        print_flags(q, "CGA split",print_flag,2) #[B, 4, 96, 160]
        
        scores = q@k.transpose(-2,-1) / gra_sharp
        print_flags(scores, "CGA q@k",print_flag,2) #[B, 4, 96, 96]
        
        scores= self.drop(F.softmax(scores, dim=1))
        h = scores@v #[B, 4, 96, 160]
        print_flags(h, "CGA scores@v",print_flag,2)
        
        h = h.transpose(1,2).contiguous() #[B, 96, 4, 160]
        print_flags(h, "CGA transpose",print_flag,2)
        
        h = merge_last(h,2) #[B, 96, 640]
        print_flags(h, "CGA merge",print_flag,2)
        
        h = h.transpose(-2,-1) #[B, 640, 96]
        print_flags(h, "CGA transpose",print_flag,2)
        
        self.scores = scores
        return h, scores

def print_flags(x,name,flag,tab_flag):
    if flag == True:
        print('\t' * tab_flag + f'{name} : {x.shape}')

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        print_flags(x,"MUHSA_TDC_B,P,C",print_flag,2)
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        print_flags(x,"MUHSA_TDC_transpose",print_flag,2)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        print_flags(q,"MUHSA_TDC_proj_qkv",print_flag,2)
        q = q.flatten(2)  # [B, 4*4*40, dim]
        print_flags(q,"MUHSA_TDC_q_flatten",print_flag,2)
        q = q.transpose(1,2)
        print_flags(q,"MUHSA_TDC_q_transpose",print_flag,2)
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        print_flags(x,"MUHSA_TDC_flatten, transpose",print_flag,2)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        print_flags(q,"MUHSA_TDC_split_last",print_flag,2)
        
        if Test == 'Test1':
            q, k, v = [x.transpose(-2, -1) for x in [q,k,v]]
            print_flags(q,"(Test1)MUHSA_TDC_transpose",print_flag,2)
        
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp
        print_flags(scores,"MUHSA_TDC_Q@Ke",print_flag,2)

        scores = self.drop(F.softmax(scores, dim=-1))
        print_flags(scores,"MUHSA_TDC_Q@K_drop",print_flag,2)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        if Test == 'Test1':
            h = (scores @ v)
            print_flags(h,"(Test1)MUHSA_TDC_Scores@V",print_flag,2)
            h = h.transpose(1,2).contiguous()
            print_flags(h,"(Test1)MUHSA_TDC_transpose",print_flag,2)
            h = h.transpose(1,3).contiguous()
            print_flags(h,"(Test1)MUHSA_TDC_transpose",print_flag,2)            
        else:
            h = (scores @ v).transpose(1, 2).contiguous()
            print_flags(h,"MUHSA_TDC_Scores@V",print_flag,2)
        # -merge-> (B, S, D)
        
        # if Test == 'Test1':
        #     h = h.transpose(-2,-1)
        #     print_flags(h,"(Test1)MUHSA_TDC_transpose",print_flag,2)       
        
        
        h = merge_last(h, 2)
        print_flags(h,"MUHSA_TDC_merge_last",print_flag,2)
        
        self.scores = scores

        return h, scores




class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        print_flags(x,"PWST_B,P,C",print_flag,2)
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        print_flags(x,"PWST_Transpose",print_flag,2)
        x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
        print_flags(x,"PWST_fc1",print_flag,2)
        x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        print_flags(x,"PWST_STConv",print_flag,2)
        x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
        print_flags(x,"PWST_fc2",print_flag,2)
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        print_flags(x,"PWST_flatten",print_flag,2)
        
        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.attn2 = ChannelGroupAttention(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        print_flags(x,"Block_ST_TDC_First",print_flag,1)
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        print_flags(Atten,"Transforemr",print_flag,1)
        h = self.drop(self.proj(Atten))
        print_flags(h,"After proj and drop",print_flag,1)
        x = x + h
        print_flags(x,"Add",print_flag,1)
        
        if Test == 'Test2':
            Atten, Score = self.attn2(self.norm1(x), gra_sharp)
            print_flags(Atten,"(Test2)Channel Attention",print_flag,1)
            h = self.drop(self.proj(Atten))
            print_flags(h,"(Test2)After proj and drop",print_flag,1)
            x = x + h
            print_flags(x,"(Test2)Add",print_flag,1)
            
        
        h = self.drop(self.pwff(self.norm2(x)))
        print_flags(h,"drop",print_flag,1)
        x = x + h
        print_flags(x,"Add2",print_flag,1)
        return x, Score

class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        i = 0
        for block in self.blocks:
            if print_flag == True:
                print(f'    num_layer : {i}')
            x, Score = block(x, gra_sharp)
            i = i+1
        return x, Score

# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
    ):
        super().__init__()

        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        #self.patch_embedding = nn.Conv3d(64, 64, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        
        
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            #nn.Conv3d(3,16, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            #nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            #nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            #nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            #nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
           
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp):

        # b is batch number, c channels, t frame, fh frame height, and fw frame width
        
        x = x.transpose(1, 2)  # t와 c의 순서를 바꿈. [B, 3, 160, 128, 128]
        print_flags(x,"After transpose",print_flag,0)
        
        b, c, t, fh, fw = x.shape
        #print(x.shape)
        
        x = self.Stem0(x)  # 출력: [B, dim//4, 160, 64, 64]
        print_flags(x,"After Stem0",print_flag,0)
        x = self.Stem1(x)  # 출력: [B, dim//2, 160, 32, 32]
        print_flags(x,"After Stem1",print_flag,0)
        x = self.Stem2(x)  # 출력: [B, dim, 160, 16, 16] / # [B, 64, 160, 64, 64]
        print_flags(x,"After Stem2",print_flag,0)
        
        x = self.patch_embedding(x)  # [B, dim, 40, 4, 4] /# [B, 64, 40, 4, 4]
        print_flags(x,"After patch_embedding",print_flag,0)
        x = x.flatten(2).transpose(1, 2)  # [B, 40*4*4, dim] /# [B, 40*4*4, 64]
        print_flags(x,"After flatten and transpose",print_flag,0)
        
        
        Trans_features, Score1 =  self.transformer1(x, gra_sharp)  # [B, 4*4*40, 64]
        print_flags(Trans_features,"After transformer1",print_flag,0)
        Trans_features2, Score2 =  self.transformer2(Trans_features, gra_sharp)  # [B, 4*4*40, 64]
        print_flags(Trans_features2,"After transformer2",print_flag,0)
        Trans_features3, Score3 =  self.transformer3(Trans_features2, gra_sharp)  # [B, 4*4*40, 64]
        print_flags(Trans_features3,"After transformer3",print_flag,0)
        
        # upsampling heads
        #features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 64, 40, 4, 4]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//4, 4, 4) # [B, 64, 40, 4, 4]
        print_flags(features_last,"After transpose and view",print_flag,0)
        
        features_last = self.upsample(features_last)		    # x [B, 64, 7*7, 80]
        print_flags(features_last,"After upsample",print_flag,0)
        
        features_last = self.upsample2(features_last)		    # x [B, 32, 7*7, 160]
        print_flags(features_last,"After upsample2",print_flag,0)
        
        
        features_last = torch.mean(features_last,3)     # x [B, 32, 160, 4]  
        print_flags(features_last,"After mean",print_flag,0)
        features_last = torch.mean(features_last,3)     # x [B, 32, 160]  
        print_flags(features_last,"After mean2",print_flag,0)
        rPPG = self.ConvBlockLast(features_last)    # x [B, 1, 160]
        print_flags(rPPG,"last",print_flag,0)
        
        rPPG = rPPG.squeeze(1)
        print_flags(rPPG,"last",print_flag,0)
        
        return rPPG, Score1, Score2, Score3
