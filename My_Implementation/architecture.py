# from _typeshed import Incomplete
import torch 
import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt 
from layer_def import * 




def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u -1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max = b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., gelu='none') -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU(approximate=gelu)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=False) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.mask = mask
        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)
        
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None
    
    def get_attn(self):
        return self.attn
    
    def save_attn(self, attn):
        self.attn = attn
    def save_attn_cam(self, cam):
        self.attn_cam = cam
    def get_attn_cam(self):
        return self.attn_cam
    def get_v(self):
        return self.v
    def save_v(self, v):
        self.v = v
    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, c, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        
        self.save_v(v)
        dots = self.matmul1([q, k]) * self.scale
        if self.mask:
            mask = torch.tril(torch.ones(n, n, device=x.device)).bool()
            dots = dots.masked_fill(~mask, float('-inf'))
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)
        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)
        
        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)
        if self.mask:
            n = cam1.shape[-1] # Get sequence length
            mask = torch.tril(torch.ones(n, n, device=cam.device)).bool()
            cam1 = cam1 * mask
        # A = Q *K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-12)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = LayerNorm(dim, eps=1e-12)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam


class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim = 768):
        super().__init__()
        img_size  = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
            (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)

class VitTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=True,mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.norm = LayerNorm(embed_dim, eps=1e-12)
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()
        self.inp_grad = None
        
    def save_inp_grag(self, grad):
        self.inp_grad = grad
    def get_inp_grad(self, grad):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x = self.add([x, self.pos_embed])
        x.register_hook(self.save_inp_grag)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        cam = self.head.relprop(cam, **kwargs)
        # print("conservation 1", cam.sum())
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)
        # print("conservation 2", cam.sum())
        
        if method == "transformer_attribution":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam 
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam

        elif method == "rollout":
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1: ]
            return cam

        elif method == "full":
            cam, _ = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            cam = cam.sum(dim=1)
            return cam
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

    def relprop_from_features(self, cam, method="transformer_attribution", **kwargs):
        cam = self.norm.relprop(cam, **kwargs)
        for i, blk in enumerate(reversed(self.blocks)):
            cam = blk.relprop(cam, **kwargs)
        (cam, pos_embed_cam) = self.add.relprop(cam, **kwargs)
        cam = cam[:, 1:]
        cam = self.patch_embed.relprop(cam, **kwargs)
        return (cam.sum(dim=1) if cam.dim() > 1 else cam, pos_embed_cam)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads, self.scale = num_heads, (dim // num_heads)**-0.5
        self.q_proj, self.kv_proj = Linear(dim, dim, bias=qkv_bias), Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop, self.proj, self.proj_drop = Dropout(attn_drop), Linear(dim, dim), Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)
        self.matmul1 = einsum('bhid,bhjd->bhij')
        self.matmul2 = einsum('bhij,bhjd->bhid')
        self.attn_cam = self.attn_gradients = None
    def save_attn_cam(self, cam): self.attn_cam = cam
    def get_attn_cam(self): return self.attn_cam
    def save_attn_gradients(self, attn_gradients): self.attn_gradients = attn_gradients
    def get_attn_gradients(self): return self.attn_gradients
    def forward(self, x, context):
        B, N_q, C = x.shape; N_kv = context.shape[1]
        q = self.q_proj(x).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        dots = self.matmul1([q,k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        attn.register_hook(self.save_attn_gradients)
        x = self.matmul2([attn,v]).transpose(1,2).reshape(B, N_q, C)
        return self.proj_drop(self.proj(x))
    
    def relprop(self, cam, **kwargs):
        cam = self.proj.relprop(self.proj_drop.relprop(cam, **kwargs), **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)
        (cam_attn, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam_attn /= 2; cam_v /= 2
        self.save_attn_cam(cam_attn)
        cam_attn = self.softmax.relprop(self.attn_drop.relprop(cam_attn, **kwargs), **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam_attn, **kwargs)
        cam_q /= 2; cam_k /= 2;
        cam_q = rearrange(cam_q, 'b h n d -> b n (h d)', h=self.num_heads)
        cam_x = self.q_proj.relprop(cam_q, **kwargs)
        cam_kv = rearrange([cam_k, cam_v], 'kv b h n d -> b n (kv h d)', kv=2, h=self.num_heads)
        cam_context = self.kv_proj.relprop(cam_kv, **kwargs)
        return cam_x, cam_context

import torch
import torch.nn as nn
# Assuming your LRP layers (einsum, Softmax, Dropout, Linear) are available from layer_def.py

class GPT2Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., mask=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.embed_dim = dim
        self.scale = self.head_dim ** -0.5
        self.mask = mask

        # LRP-enabled layers from your framework
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)
        
        # Using einsum for matrix multiplication is fine, but torch.matmul is also clear
        self.matmul1 = einsum('bhid,bhjd->bhij') # For Q @ K.T
        self.matmul2 = einsum('bhij,bhjd->bhid') # For Attn @ V

        # Placeholders for storing intermediate values for LRP
        self.attn_cam = self.attn = self.v = self.v_cam = self.attn_gradients = None
    
    # --- All your save/get methods remain the same ---
    def get_attn(self): return self.attn
    def save_attn(self, attn): self.attn = attn
    def save_attn_cam(self, cam): self.attn_cam = cam
    def get_attn_cam(self): return self.attn_cam
    def get_v(self): return self.v
    def save_v(self, v): self.v = v
    def save_v_cam(self, cam): self.v_cam = cam
    def get_v_cam(self): return self.v_cam
    def save_attn_gradients(self, attn_gradients): self.attn_gradients = attn_gradients
    def get_attn_gradients(self): return self.attn_gradients
    
    def forward(self, x):
        B, N, C = x.shape
        
        # 1. Project to a combined QKV tensor
        qkv = self.qkv(x)
        
        # 2. Split into separate Q, K, V tensors
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # 3. Reshape and permute each tensor to split into heads
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        self.save_v(v)

        # 4. Calculate attention scores
        dots = self.matmul1([q, k]) * self.scale
        
        if self.mask:
            mask = torch.tril(torch.ones(N, N, device=x.device)).bool()
            dots = dots.masked_fill(~mask, float('-inf'))
            
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        self.save_attn(attn)
        # attn.register_hook(self.save_attn_gradients) # Keep this commented if not needed
        
        # 5. Apply attention to V
        out = self.matmul2([attn, v])
        
        # 6. Merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        # 7. Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        # 7. Invert final projection and dropout
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)

        # 6. Invert head merging
        B, N, C = cam.shape
        cam = cam.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 5. Invert attn @ v
        (cam_attn, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam_attn /= 2
        cam_v /= 2
        self.save_v_cam(cam_v)
        self.save_attn_cam(cam_attn)
        
        # Invert dropout, softmax, and causal mask
        cam_attn = self.attn_drop.relprop(cam_attn, **kwargs)
        cam_attn = self.softmax.relprop(cam_attn, **kwargs)
        if self.mask:
            seq_len = cam_attn.shape[-1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=cam.device)).bool()
            cam_attn = cam_attn * mask

        # 4. Invert q @ k.T
        (cam_q, cam_k) = self.matmul1.relprop(cam_attn, **kwargs)
        cam_q /= 2
        cam_k /= 2

        # 3. Invert head splitting for each of q, k, v
        cam_q = cam_q.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        cam_k = cam_k.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        cam_v = cam_v.permute(0, 2, 1, 3).contiguous().view(B, N, C)

        # 2. Invert the .split() operation by concatenating
        cam_qkv = torch.cat([cam_q, cam_k, cam_v], dim=2)

        # 1. Invert the initial qkv projection
        return self.qkv.relprop(cam_qkv, **kwargs)
    
class GPT2DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1, self.attn = LayerNorm(dim, eps=1e-5), GPT2Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mask=True)
        self.norm2, self.cross_attn = LayerNorm(dim, eps=1e-5), CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm3, self.mlp = LayerNorm(dim, eps=1e-5), Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop, gelu='tanh')
        self.add1, self.add2, self.add3 = Add(), Add(), Add()
        self.clone1, self.clone2, self.clone3 = Clone(), Clone(), Clone()

    # def forward(self, x, context):
    #     # Standard Pre-LN Self-Attention with a simple residual connection
    #     x = x + self.attn(self.norm1(x))
        
    #     # Standard Pre-LN Cross-Attention with a simple residual connection
    #     x = x + self.cross_attn(self.norm3(x), context)
        
    #     # Standard Pre-LN MLP with a simple residual connection
    #     x = x + self.mlp(self.norm2(x))
        
    #     return x
    def forward(self, x, context):

        # x = x + self.attn(self.norm1(x))
        # x = x + self.cross_attn(self.norm2(x), context)
        # x = x + self.mlp(self.norm3(x))
        x1, x2 = self.clone1(x, 2); x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2); x = self.add2([x1, self.cross_attn(self.norm3(x2), context)])
        x1, x2 = self.clone3(x, 2); x = self.add3([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, context_cam, **kwargs):
        (cam1, cam2) = self.add3.relprop(cam, **kwargs);cam2 = self.mlp.relprop(self.norm2.relprop(cam2, **kwargs), **kwargs); cam = self.clone3.relprop((cam1, cam2), **kwargs)
        (cam1, cam2) = self.add2.relprop(cam, **kwargs); cam2, new_context_cam = self.cross_attn.relprop(cam2, **kwargs); cam2 = self.norm3.relprop(cam2, **kwargs); cam = self.clone2.relprop((cam1, cam2), **kwargs)
        context_cam += new_context_cam
        (cam1, cam2) = self.add1.relprop(cam, **kwargs); cam2 = self.attn.relprop(self.norm1.relprop(cam2, **kwargs), **kwargs); cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam, context_cam
    

    

class VitGPT2Model(nn.Module):
    def __init__(self, encoder, vocab_size=50257, max_seq_len=1024, embed_dim=768, depth=12, num_heads=12,drop=.1):
        super().__init__()
        self.encoder, self.embed_dim = encoder, embed_dim
        self.decoder_embed = TokenEncoder(vocab_size, embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.decoder_blocks = nn.ModuleList([GPT2DecoderBlock(dim=embed_dim, num_heads=num_heads,drop=drop,attn_drop=drop) for _ in range(depth)])
        self.decoder_norm, self.lm_head = LayerNorm(embed_dim, eps=1e-5), Linear(embed_dim, vocab_size, bias=False)
        self.add = Add()
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
    def forward(self, image, caption):

        image_features = self.encoder.forward_features(image)
        token_embeddings = self.decoder_embed(caption)
        pos_embeddings = self.decoder_pos_embed[:, :caption.shape[1], :]
        x = self.add([token_embeddings, pos_embeddings])
        for blk in self.decoder_blocks:
            x = blk(x, image_features)
        x = self.decoder_norm(x)
        output_logits = self.lm_head(x)
        return output_logits
    
    def relprop(self, cam, method='transformer_attribution', start_layer=0,initial_logit=None,logging = False, **kwargs):
        if method == "transformer_attribution":
            _cam = self.lm_head.relprop(cam.clone(), **kwargs)
            _cam = self.decoder_norm.relprop(_cam, **kwargs)
            num_patches = self.encoder.patch_embed.num_patches
            _context_cam = torch.zeros(cam.shape[0], num_patches+1, self.embed_dim).to(cam.device)
            for blk in reversed(self.decoder_blocks):
                _cam, _context_cam = blk.relprop(_cam, _context_cam, **kwargs)
            (_cam, _) = self.add.relprop(_cam, **kwargs)
            text_cam = self.decoder_embed.relprop(_cam, **kwargs)
            self.encoder.relprop_from_features(_context_cam, **kwargs)
            encoder_attributions = [(blk.attn.get_attn_gradients()[0] * blk.attn.get_attn_cam()[0]).clamp(min=0).mean(dim=0).unsqueeze(0) for blk in self.encoder.blocks]
            encoder_rollout = compute_rollout_attention(encoder_attributions, start_layer=start_layer)
            target_token_idx = cam.shape[1] - 1
            cross_attribution_sum = torch.zeros(1,1,num_patches + 1).to(cam.device)
            for blk in self.decoder_blocks:
                grad = blk.cross_attn.get_attn_gradients()[0].reshape(blk.cross_attn.num_heads, -1, num_patches + 1)[:, target_token_idx, :].unsqueeze(1)
                cam_ = blk.cross_attn.get_attn_cam()[0].reshape(blk.cross_attn.num_heads, -1, num_patches + 1)[:, target_token_idx, :].unsqueeze(1)
                attr = (grad * cam_).clamp(min=0).mean(dim=0)
                cross_attribution_sum += attr
            final_cam = (cross_attribution_sum @ encoder_rollout).squeeze(0)
            return final_cam[:, 1:], text_cam
        elif method == 'full':
            
            cam = self.lm_head.relprop(cam,**kwargs)
            # print(f"After LM Head           {cam.sum().item():.4f}")
            
            cam = self.decoder_norm.relprop(cam, **kwargs)
            # print(f"After Decoder Norm: {cam.sum().item():.4f}")
            num_patches = self.encoder.patch_embed.num_patches
            context_cam = torch.zeros(cam.shape[0], num_patches + 1, self.embed_dim).to(cam.device)
            for i, blk in enumerate(reversed(self.decoder_blocks)): 
                cam, context_cam = blk.relprop(cam, context_cam, **kwargs)
                # print(f"After Decoder Block {len(self.decoder_blocks) - i - 1}: {(cam.sum() + context_cam.sum()).item():.4f} (Text: {cam.sum().item():.4f}, Image: {context_cam.sum().item():.4f})")
            (cam, positional_cam) = self.add.relprop(cam, **kwargs)
            text_sum, text_pos_sum = cam.sum().item(), positional_cam.sum().item()
            # print(f"After Embed+Pos Add:  Embedding: {text_sum:.4f}, Position: {text_pos_sum:.4f}")
            _ = self.decoder_embed.relprop(cam, **kwargs)
            # print(f"Relevance sent to Encoder: {context_cam.sum().item():.4f}")
            final_image_map, encoder_pos_cam = self.encoder.relprop_from_features(context_cam, **kwargs)
            image_sum, image_pos_sum = final_image_map.sum().item(), encoder_pos_cam.sum().item()
            if logging:
                print(f"Total Attribution is split along 4 parts: \nText LRP: {text_sum:.4f} \nText Pos: {text_pos_sum:.4f}\nImage LRP: {image_sum:.4f} \nImage Pos: {image_pos_sum:.4f}")
                print(f"----------------------------------------")
                print(f"Total: {text_pos_sum + text_sum + image_pos_sum + image_sum:.5f}")
            
            return final_image_map, cam
        

# def _conv_filter(state_dict, patch_size=16):
#     out_dict = {}
#     for k,v in state_dict.items():
#         if 'patch_embed.proj.weight' in k:
#             v = v.reshape((v.shape[0], 3, patch_size, patch_size))
#         out_dict[k] = v
#     return out_dict

# def vit_base_patch16_224(pretrained=True, **kwargs):
#     model = VitTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs
#     )
#     model.default_cfg = default_cfgs['vit_base_patch16_224']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter
#         )
#         return model


class LRP:
    
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, caption, target_index=None, method='full', start_layer=0):
        output = self.model(input, caption)
        last_token_logits = output[:, -1, :]
        target_index = last_token_logits.argmax(dim=-1).item()
        one_hot = torch.zeros_like(last_token_logits)
        one_hot[:, target_index] = 1
        full_one_hot = torch.zeros_like(output)
        full_one_hot[:, -1, :] = one_hot
        self.model.zero_grad()
        output.backward(gradient=full_one_hot, retain_graph=True)
        return self.model.relprop(full_one_hot, method=method, start_layer=start_layer, logging=True, alpha=1)
    

class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]
    
    
def convert_hf_to_custom(source_state_dict, target_model):
    """
    Loads weights from a Hugging Face ViT-GPT2 state_dict into a custom model.
    
    Args:
        source_state_dict (dict): The state dictionary from the pre-trained model.
        target_model (nn.Module): An instance of your custom model.
    
    Returns:
        A new state dictionary compatible with the target model.
    """
    target_state_dict = target_model.state_dict()
    new_state_dict = OrderedDict()

    print("Starting weight conversion...")

    for target_key, target_tensor in target_state_dict.items():
        # --- ViT Encoder Mapping ---
        if 'encoder.patch_embed.proj' in target_key:
            source_key = target_key.replace('encoder.patch_embed.proj', 'vit.embeddings.patch_embeddings.projection')
            new_state_dict[target_key] = source_state_dict[source_key]

        elif 'encoder.pos_embed' in target_key:
            source_key = 'vit.embeddings.position_embeddings'
            new_state_dict[target_key] = source_state_dict[source_key]
        elif 'encoder.cls_token' in target_key:
            source_key = 'vit.embeddings.cls_token'
            new_state_dict[target_key] = source_state_dict[source_key]
            
        elif 'encoder.blocks' in target_key:
            parts = target_key.split('.')
            block_idx = parts[2]
            
            # Map Norms
            if 'norm1' in target_key:
                source_key = f"vit.encoder.layer.{block_idx}.layernorm_before.{parts[-1]}"
                new_state_dict[target_key] = source_state_dict[source_key]
            elif 'norm2' in target_key:
                source_key = f"vit.encoder.layer.{block_idx}.layernorm_after.{parts[-1]}"
                new_state_dict[target_key] = source_state_dict[source_key]
            
            # Map Attention Projection
            elif 'attn.proj' in target_key:
                source_key = f"vit.encoder.layer.{block_idx}.attention.output.dense.{parts[-1]}"
                new_state_dict[target_key] = source_state_dict[source_key]

            # Map MLP Layers
            elif 'mlp.fc1' in target_key:
                source_key = f"vit.encoder.layer.{block_idx}.intermediate.dense.{parts[-1]}"
                new_state_dict[target_key] = source_state_dict[source_key]
            elif 'mlp.fc2' in target_key:
                source_key = f"vit.encoder.layer.{block_idx}.output.dense.{parts[-1]}"
                new_state_dict[target_key] = source_state_dict[source_key]

            # SPECIAL CASE: Combine Q, K, V into one QKV layer
            elif 'attn.qkv.weight' in target_key:
                q_w = source_state_dict[f'vit.encoder.layer.{block_idx}.attention.attention.query.weight']
                k_w = source_state_dict[f'vit.encoder.layer.{block_idx}.attention.attention.key.weight']
                v_w = source_state_dict[f'vit.encoder.layer.{block_idx}.attention.attention.value.weight']
                new_state_dict[target_key] = torch.cat([q_w, k_w, v_w], dim=0)

            elif 'attn.qkv.bias' in target_key:
                q_b = source_state_dict[f'vit.encoder.layer.{block_idx}.attention.attention.query.bias']
                k_b = source_state_dict[f'vit.encoder.layer.{block_idx}.attention.attention.key.bias']
                v_b = source_state_dict[f'vit.encoder.layer.{block_idx}.attention.attention.value.bias']
                new_state_dict[target_key] = torch.cat([q_b, k_b, v_b], dim=0)

        elif 'encoder.norm' in target_key:
            source_key = target_key.replace('encoder.norm', 'vit.layernorm')
            new_state_dict[target_key] = source_state_dict[source_key]

        # --- GPT2 Decoder Mapping ---
        # This assumes your 'decoder_embed' is a direct mapping from gpt2's wte
        elif 'decoder_embed.linear.weight' in target_key: # Or whatever you name it
            source_key = 'gpt2.transformer.wte.weight'
            if source_state_dict[source_key].T.shape == target_tensor.shape:
                new_state_dict[target_key] = source_state_dict[source_key].T
            else:
                print(source_state_dict[source_key].shape, target_tensor.shape)
                print(f"!! Vocab size mismatch for {target_key}. Skipping.")

        # Mapping position embeddings
        elif 'decoder_pos_embed' in target_key:
            source_key = 'gpt2.transformer.wpe.weight'
            # FIX: Add a batch dimension to the source tensor
            if source_state_dict[source_key].unsqueeze(0).shape == target_tensor.shape:
                new_state_dict[target_key] = source_state_dict[source_key].unsqueeze(0)
            else:
                print(f"!! Shape mismatch for {target_key}. Skipping.")
            
        elif 'decoder_blocks' in target_key:
            parts = target_key.split('.')
            block_idx = parts[1]
            
            # Map Norms
            if 'norm1' in target_key: source_key = f"gpt2.transformer.h.{block_idx}.ln_1.{parts[-1]}"
            elif 'norm2' in target_key: source_key = f"gpt2.transformer.h.{block_idx}.ln_2.{parts[-1]}"
            elif 'norm3' in target_key: source_key = f"gpt2.transformer.h.{block_idx}.ln_cross_attn.{parts[-1]}"
            else: source_key = None
            if source_key: new_state_dict[target_key] = source_state_dict[source_key]

            # Map self-attention QKV and Proj (Conv1D -> Linear requires transpose)
            if 'attn.qkv' in target_key and 'cross' not in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.attn.c_attn"
            elif 'attn.proj' in target_key and 'cross' not in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.attn.c_proj"
            # Map cross-attention
            elif 'cross_attn.q_proj' in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.crossattention.q_attn"
            elif 'cross_attn.kv_proj' in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.crossattention.c_attn"
            elif 'cross_attn.proj' in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.crossattention.c_proj"
            # Map MLP
            elif 'mlp.fc1' in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.mlp.c_fc"
            elif 'mlp.fc2' in target_key: source_key_base = f"gpt2.transformer.h.{block_idx}.mlp.c_proj"
            else: source_key_base = None

            if source_key_base:
                source_param = 'weight' if 'weight' in target_key else 'bias'
                source_key = f"{source_key_base}.{source_param}"
                
                # Conv1D weights need to be transposed
                if 'weight' in source_key:
                    new_state_dict[target_key] = source_state_dict[source_key].T
                else: # Bias does not need transpose
                    new_state_dict[target_key] = source_state_dict[source_key]
        
        elif 'decoder_norm' in target_key:
            source_key = target_key.replace('decoder_norm', 'gpt2.transformer.ln_f')
            new_state_dict[target_key] = source_state_dict[source_key]
        
        elif 'lm_head' in target_key:
            source_key = 'gpt2.lm_head.weight'
            if source_state_dict[source_key].shape == target_tensor.shape:
                new_state_dict[target_key] = source_state_dict[source_key]
            else:
                print(f"!! Vocab size mismatch for {target_key}. Skipping.")

    # Final check for missed keys
    converted_keys = set(new_state_dict.keys())
    all_target_keys = set(target_state_dict.keys())
    missed_keys = all_target_keys - converted_keys
    if missed_keys:
        print(f"\nWarning: The following keys in the target model were not found in the source mapping:")
        for key in sorted(list(missed_keys)):
            print(f"  - {key}")

    return new_state_dict


if __name__ == '__main__':
    vit_encoder = VitTransformer()
    model = VitGPT2Model(encoder=vit_encoder, vocab_size=10000, depth=12)
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_caption = torch.randint(0, 10000, (1, 10))
    lrp_generator = LRP(model)
    attribution_full, _ = lrp_generator.generate_LRP(dummy_image, dummy_caption, method='full')
    print(f"Shape of 'full' attribution map: {attribution_full.shape}")
    print("\n--- Generating with 'transformer_attribution' method ---")
    attribution_transformer = lrp_generator.generate_LRP(dummy_image, dummy_caption, method='transformer_attribution')
    print(f"Shape of 'transformer_attribution' map: {attribution_transformer.shape}")
    





