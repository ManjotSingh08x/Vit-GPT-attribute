from typing import Tuple, OrderedDict
import torch
import torch.nn as nn 
import torch.nn.functional as F

def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True
    self.Y = output

def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output

def safe_divide(a, b):
    den = b
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / (den)

class RelProp(nn.Module):
    def __init__(self) -> None:
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    # S is upstream gradient flow wrt to Z. Z is a function of X
    # C is gradient flow wrt to X now. 
    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z,X,S, retain_graph=True)
        return C

    # Forwards and does nothing
    def relprop(self, R, alpha):
        return R

class RelPropSimple(RelProp):
    # General Deep Taylor Decomposition
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        
        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

class AddEye(RelPropSimple):
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)

class ReLU(nn.ReLU, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass

class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


# Understood now. Normalization of Addition operation
class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)
    
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]
        return outputs
    
class Cat(RelProp):
    def forward(self, inputs, dim):
        self.dim = dim
        return torch.cat(inputs, dim)
    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        return [ x * c for x, c in zip(self.X, C)]

class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs
    
    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]
        R = self.X * C
        return R
# Cat, Batchnorm

class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R

# BatchNorm2D

class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min = 0)
        nw = torch.clamp(self.weight, max = 0)
        px = torch.clamp(self.X, min = 0)
        nx = torch.clamp(self.X, max = 0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2
        
        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

# Conv2D
class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2
            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

class MultiheadAttention(RelProp):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.softmax = Softmax(dim=-1)

        self.einsum1 = einsum('bid,bjd->bij')
        self.einsum2 = einsum('bij,bjd->bid')

        self._register_load_state_dict_pre_hook(MultiheadAttention._pre_load_state_dict)

        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn(self, attn):
        self.attn = attn

    def get_attn(self):
        return self.attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    @staticmethod
    def _pre_load_state_dict(state_dict: OrderedDict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        w = state_dict[prefix + 'in_proj_weight']
        b = state_dict[prefix + 'in_proj_bias']

        embed_dim = w.shape[1]

        state_dict[prefix + 'q_proj.weight'] = w[:embed_dim]
        state_dict[prefix + 'q_proj.bias'] = b[:embed_dim]

        state_dict[prefix + 'k_proj.weight'] = w[embed_dim:2*embed_dim]
        state_dict[prefix + 'k_proj.bias'] = b[embed_dim:2*embed_dim]

        state_dict[prefix + 'v_proj.weight'] = w[2*embed_dim:]
        state_dict[prefix + 'v_proj.bias'] = b[2*embed_dim:]

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len, _, _ = key.size()

        self.tgt_len = tgt_len
        self.src_len = src_len
        self.bsz = bsz

        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        self.head_dim = head_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)  # BHxSxD

        # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = self.einsum1([q, k])  # BHxTxS

        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.dropout(attn_output_weights)

        self.save_attn(attn_output_weights)
        attn_output_weights.register_hook(self.save_attn_gradients)

        # attn_output = torch.bmm(attn_output_weights, v)
        attn_output = self.einsum2([attn_output_weights, v])  # BHxTxD

        #  BHxTxD -> TxBHxD -> TxBxHD
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

    def relprop(self, cam_attn_output, alpha, **kwargs):
        cam_attn_output = self.out_proj.relprop(cam_attn_output, alpha, **kwargs)
        cam_attn_output = cam_attn_output.view(self.tgt_len, self.bsz*self.num_heads, self.head_dim).transpose(0, 1)
        cam_attn_output_weights, cam_v = self.einsum2.relprop(cam_attn_output, alpha, **kwargs)
        cam_attn_output_weights /= 2
        cam_v /= 2
        self.save_attn_cam(cam_attn_output_weights)
        cam_attn_output_weights = self.dropout.relprop(cam_attn_output_weights, alpha, **kwargs)
        cam_attn_output_weights = self.softmax.relprop(cam_attn_output_weights, alpha, **kwargs)
        cam_q, cam_k = self.einsum1.relprop(cam_attn_output_weights, alpha, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_v = cam_v.transpose(0, 1).view(self.src_len, self.bsz, self.num_heads*self.head_dim)
        cam_k = cam_k.transpose(0, 1).view(self.src_len, self.bsz, self.num_heads*self.head_dim)
        cam_q = cam_q.transpose(0, 1).view(self.tgt_len, self.bsz, self.num_heads*self.head_dim)

        pre_cam_v = cam_v.min() == cam_v.max() == 0
        cam_v = self.v_proj.relprop(cam_v, alpha, **kwargs)
        cam_k = self.k_proj.relprop(cam_k, alpha, **kwargs)
        cam_q = self.q_proj.relprop(cam_q, alpha, **kwargs)

        if cam_v.min() == cam_v.max() == 0 and not pre_cam_v:
            cam_k_sum = cam_k.sum()
            cam_q_sum = cam_q.sum()
            cam_k_fact = safe_divide(cam_k_sum.abs(), cam_k_sum.abs() + cam_q_sum.abs()) * cam_attn_output.sum()
            cam_q_fact = safe_divide(cam_q_sum.abs(), cam_k_sum.abs() + cam_q_sum.abs()) * cam_attn_output.sum()

            cam_k = cam_k * safe_divide(cam_k_fact, cam_k.sum())
            cam_q = cam_q * safe_divide(cam_q_fact, cam_q.sum())

        return cam_q, cam_k, cam_v

class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = Linear(vocab_size, embed_dim, bias = False)
    def forward(self, x):
        one_hot = F.one_hot(x, num_classes=self.vocab_size).float()
        return self.linear(one_hot)
    def relprop(self, R, alpha):
        return self.linear.relprop(R, alpha)