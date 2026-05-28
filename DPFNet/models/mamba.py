import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.function import DropPath, Mlp, Identity
import numpy as np

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)

class Mamba(nn.Layer):
    def __init__(
        self, 
        depth=12,
        d_inner=512,
        d_model=64,
        state_size=16,
        num_classes=8,
        init_scale=0.001,
        norm_layer=nn.LayerNorm
    ):
        super(Mamba, self).__init__()
        self.blocks = nn.LayerList(
            [ResidualBlock(d_inner, d_model, state_size) for i in range(depth)]
        )

    def forward(self, x, y=None):
        x_o = x.clone()
        for blk in self.blocks:
            if y is not None:
                x, x_o = blk(x, x_o, y)
            else:
                x = blk(x)
        return x

class ResidualBlock(nn.Layer):
    def __init__(self, d_inner, d_model, state_size):
        super().__init__()

        self.mambablock = MambaBlock(d_inner, d_model, state_size)
        self.epsilon = 1e-6
        self.norm_weight = paddle.create_parameter(
            shape=[d_inner],
            dtype=paddle.float32,
            default_initializer=ones_
        )
        self.norm_bias = paddle.create_parameter(
            shape=[d_inner],
            dtype=paddle.float32,
            default_initializer=zeros_
        )

    def forward(self, x, x_o, y): 
        output = paddle.incubate.nn.functional.fused_rms_norm(x, self.norm_weight, self.norm_bias, self.epsilon, 2)
        output, x_o = self.mambablock(output[0], x_o, y)
        output = output + x
        return output, x_o

class MambaBlock(nn.Layer):
    def __init__(self, d_inner, d_model, state_size):
        super().__init__()

        self.in_proj = nn.Linear(d_inner, 2 * d_inner, bias_attr=False)

        self.conv1d = nn.Conv1D(d_inner, d_inner, kernel_size=4, padding=3)

        self.ssm = S6(d_inner, d_model, state_size)

    def forward(self, x, x_o, y):

        batch_size, seq_len = x.shape[:2]

        xz = self.in_proj(x)
        x, z = paddle.chunk(xz, chunks=2, axis=-1)

        # x branch
        x = x.transpose([0, 2, 1])
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose([0, 2, 1])

        x = F.silu(x)
        x, x_o = self.ssm(x, x_o, y)

        # z branch
        z = F.silu(z)

        output = x * z
        # output = self.out_proj(output)

        return output, x_o

class CrossAttention(nn.Layer):
    """Cross Attention Layer"""

    def __init__(
        self,
        dim,
        num_heads=8,
        output_dim=512,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
        add_self=False
    ):
        super().__init__()
        self.add_self = add_self
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias_attr=False)
        self.kv = nn.Linear(dim, all_head_dim * 2, bias_attr=False)

        if qkv_bias:
            self.q_bias = paddle.create_parameter(
                shape=[all_head_dim], dtype="float32", default_initializer=nn.initializer.Constant(0.0)
            )
            self.v_bias = paddle.create_parameter(
                shape=[all_head_dim], dtype="float32", default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, output_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B, N, C = key_value.shape
        q_bias = None
        kv_bias = None
        if self.q_bias is not None:
            q_bias = self.q_bias
            kv_bias = paddle.concat(
                (paddle.zeros_like(self.v_bias), self.v_bias)
            )

        q = F.linear(x=query, weight=self.q.weight, bias=q_bias)
        kv = F.linear(x=key_value, weight=self.kv.weight, bias=kv_bias)
        kv_q = F.linear(x=query, weight=self.kv.weight, bias=kv_bias)

        q = q.reshape([q.shape[0], q.shape[1], self.num_heads, -1]).transpose([0, 2, 1, 3])
        k, v = kv.reshape([B, N, 2, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])[0], kv.reshape([B, N, 2, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])[1]
        qk, qv = kv_q.reshape([B, q.shape[2], 2, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])[0], kv_q.reshape([B, q.shape[2], 2, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])[1]
        q = q * self.scale

        attn = q @ k.transpose([0, 1, 3, 2])
        if self.add_self:
            self_attn = paddle.einsum('bhnc,bhnc->bhn', q, qk)
            attn = paddle.concat([self_attn.unsqueeze(-1), attn], axis=-1)
            attn = F.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            attn, self_attn = attn[:, :, :, 1:], attn[:, :, :, 0:1]
            x = ((attn @ v) + self_attn * qv).transpose([0, 2, 1, 3]).reshape([B, q.shape[2], -1])
        else:
            attn = F.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, q.shape[2], -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 定义S6模块
class S6(nn.Layer):
    def __init__(self, d_inner, d_model, state_size):
        super(S6, self).__init__()
        # 一系列线性变换
        self.fc1 = nn.Linear(d_inner, d_model, bias_attr=True)
        self.fc2 = nn.Linear(d_inner, state_size, bias_attr=False)
        self.fc3 = nn.Linear(d_inner, state_size, bias_attr=False)
        self.fc4 = nn.Linear(d_model, d_inner, bias_attr=False)
        self.fc5 = nn.Linear(d_inner, state_size, bias_attr=False)
        self.proj = nn.Linear(d_inner, d_model, bias_attr=False)
        self.attn = CrossAttention(d_inner, output_dim=d_inner, add_self=False)
        # 设定一些超参数
        self.d_model = d_model
        self.state_size = state_size

        # dt initialization  
        # dt weights  
        dt_init_std = math.ceil(d_model / 64)**-0.5

        dt_min = 0.001
        dt_max = 0.1 
        dt_init_floor = 1e-4

        dt = paddle.exp(  
            paddle.uniform(shape=[d_model]) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)  
        ).clip(dt_init_floor)
        
        inv_dt = dt + paddle.log1p(-paddle.exp(-dt))

        self.fc1_weights = paddle.create_parameter(
            shape=[d_inner, d_model],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-dt_init_std, dt_init_std)
        )
        self.fc1_bias = paddle.create_parameter(
            shape=inv_dt.shape,
            dtype=str(inv_dt.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(inv_dt)
        )
        A = paddle.arange(1, state_size + 1, dtype='float32').reshape([1, state_size])
        A = paddle.tile(A, [d_model, 1])
        A_log = paddle.log(A)
        self.A_log = paddle.create_parameter(
            shape=A_log.shape,
            dtype=str(A_log.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(A_log)
        )

        self.D = paddle.create_parameter(
            shape=[d_inner],
            dtype="float32",
            default_initializer=ones_,
        )
        G = paddle.arange(1, state_size + 1, dtype='float32').reshape([1, state_size])
        G = paddle.tile(G, [d_model, 1])
        G_log = paddle.log(G)
        self.G_log = paddle.create_parameter(
            shape=G_log.shape,
            dtype=str(G_log.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(G_log)
        )

        self.H = paddle.create_parameter(
            shape=[d_inner],
            dtype="float32",
            default_initializer=ones_,
        )
    def forward(self, x_in, x_o, z):
        batch_size, seq_len = x_in.shape[:2]
        x = self.fc1(x_in)
        delta = F.linear(x=x_in, weight=self.fc1_weights, bias=self.fc1_bias)
        delta = F.softplus(delta)

        A = -paddle.exp(self.A_log)
        G = -paddle.exp(self.G_log)
        B = self.fc2(x_in)
        C = self.fc3(x_in)
        H = self.fc5(x_o)
        # 离散化
        dB = paddle.einsum('bld,bln->bldn', delta, B)
        dA = paddle.exp(paddle.einsum('bld,dn->bldn', delta, A))
        BX = dB * x.unsqueeze(-1)

        x_o = x_o + self.attn(x_o, z)
        dH = paddle.einsum('bld,bln->bldn', delta, H)
        dG = paddle.exp(paddle.einsum('bld,dn->bldn', delta, G))
        HZ = dH * self.proj(x_o).unsqueeze(-1)

        h = paddle.zeros((batch_size, self.d_model, self.state_size))
        u = paddle.zeros((batch_size, self.d_model, self.state_size))
        ys = []
        for t in range(0, seq_len):
            u = dG[:, t] * u + HZ[:, t] # B, D, N
            h = dA[:, t] * h + BX[:, t] # B, D, N
            y = paddle.einsum('bdn,bn->bd', (h + u) / 2, C[:, t])
            ys.append(y)
            
        y = paddle.stack(ys, axis=1) # B, L, D, N
        y = self.fc4(y) + self.D * x_in
        return y, x_o
