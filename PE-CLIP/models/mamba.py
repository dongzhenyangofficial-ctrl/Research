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
        input_dim=512,
        d_inner=384,
        d_model=64,
        state_size=16,
        n_head=8,
        num_classes=8,
        init_scale=0.001,
        norm_layer=nn.LayerNorm
    ):
        super(Mamba, self).__init__()
        self.blocks = nn.LayerList(
            [ResidualBlock(input_dim, d_inner, d_model, state_size, n_head) for i in range(depth)]
        )

        self.fc_norm = norm_layer(input_dim) if num_classes > 0 else None
        self.head = nn.Linear(input_dim, num_classes) if num_classes > 0 else Identity()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight)
            self.head.weight.set_value(
                self.head.weight.multiply(paddle.to_tensor(init_scale))
            )
            self.head.bias.set_value(
                self.head.bias.multiply(paddle.to_tensor(init_scale))
            )

    def forward(self, x, bool_masked_pos=None):

        for blk in self.blocks:

            x = blk(x)
        
        # F = paddle.mean(x, axis=1)
        # T = paddle.to_tensor(np.load('Text_vector.npy'))
        # F /= F.norm(axis=-1, keepdim=True)
        # T /= T.norm(axis=-1, keepdim=True)
        # output = (100.0 * F @ T.t())

        # output = self.fc_norm(paddle.mean(x, axis=1))
        # output = self.head(output)
        return x

class ResidualBlock(nn.Layer):
    def __init__(self, input_dim, d_inner, d_model, state_size, n_head):
        super().__init__()

        self.mambablock = MambaBlock(input_dim, d_inner, d_model, state_size, n_head)
        self.input_dim = input_dim
        self.epsilon = 1e-6
        self.norm_weight = paddle.create_parameter(
            shape=[self.input_dim],
            dtype=paddle.float32,
            default_initializer=ones_
        )
        self.norm_bias = paddle.create_parameter(
            shape=[self.input_dim],
            dtype=paddle.float32,
            default_initializer=zeros_
        )

    def forward(self, x):
        output = paddle.incubate.nn.functional.fused_rms_norm(x, self.norm_weight, self.norm_bias, self.epsilon, 2)
        output = self.mambablock(output[0])
        output = output + x
        return output

class MambaBlock(nn.Layer):
    def __init__(self, input_dim, d_inner, d_model, state_size, n_head):
        super().__init__()

        self.in_proj = nn.Linear(input_dim, 2 * d_inner, bias_attr=False)

        self.conv1d = nn.Conv1D(d_inner, d_inner, kernel_size=4, padding=3)
        
        A = paddle.arange(1, state_size + 1, dtype='float32').reshape([1, 1, state_size])
        A = paddle.tile(A, [n_head, d_model, 1])
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

        self.ssm = S6(d_inner, d_model, state_size, n_head)

        self.out_proj = nn.Linear(d_inner, input_dim, bias_attr=False)

    def forward(self, x):

        batch_size, seq_len = x.shape[:2]

        xz = self.in_proj(x)
        x, z = paddle.chunk(xz, chunks=2, axis=-1)

        # x branch
        x = x.transpose([0, 2, 1])
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose([0, 2, 1])

        x = F.silu(x)
        y = self.ssm(x, self.A_log, self.D)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        return output
    
# 定义S6模块
class S6(nn.Layer):
    def __init__(self, d_inner, d_model, state_size, n_head):
        super(S6, self).__init__()
        # 一系列线性变换
        self.fc1 = nn.Linear(d_inner, d_model * n_head, bias_attr=True)
        self.fc2 = nn.Linear(d_inner, state_size * n_head, bias_attr=False)
        self.fc3 = nn.Linear(d_inner, state_size * n_head, bias_attr=False)
        self.fc4 = nn.Linear(d_model * n_head, d_inner, bias_attr=False)
        # 设定一些超参数
        self.d_model = d_model
        self.state_size = state_size
        self.n_head = n_head
        # dt initialization  
        # dt weights  
        dt_init_std = math.ceil(d_model / 64)**-0.5

        dt_min = 0.001
        dt_max = 0.1 
        dt_init_floor = 1e-4

        dt = paddle.exp(  
            paddle.uniform(shape=[n_head * d_model]) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)  
        ).clip(dt_init_floor)
        
        inv_dt = dt + paddle.log1p(-paddle.exp(-dt))

        self.fc1_weights = paddle.create_parameter(
            shape=[d_inner, n_head * d_model],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-dt_init_std, dt_init_std)
            # default_initializer = nn.initializer.Constant(value=dt_init_std)
        )
        self.fc1_bias = paddle.create_parameter(
            shape=inv_dt.shape,
            dtype=str(inv_dt.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(inv_dt)
        )

    def forward(self, x_in, A, D):
        batch_size, seq_len = x_in.shape[:2]
        x = self.fc1(x_in)
        delta = F.linear(x=x_in, weight=self.fc1_weights, bias=self.fc1_bias)
        delta = F.softplus(delta)

        A = -paddle.exp(A)
        B = self.fc2(x_in)
        C = self.fc3(x_in)

        delta = delta.reshape([batch_size, seq_len, self.n_head, self.d_model])
        B = B.reshape([batch_size, seq_len, self.n_head, self.state_size])
        C = C.reshape([batch_size, seq_len, self.n_head, self.state_size])
        x = x.reshape([batch_size, seq_len, self.n_head, self.d_model])
        delta = delta.transpose([0, 2, 1, 3])
        B = B.transpose([0, 2, 1, 3])
        C = C.transpose([0, 2, 1, 3])
        x = x.transpose([0, 2, 1, 3])
        # 离散化
        dB = paddle.einsum('bhld,bhln->bhldn', delta, B)
        dA = paddle.exp(paddle.einsum('bhld,hdn->bhldn', delta, A))
        BX = dB * x.unsqueeze(-1)
        
        h = paddle.zeros((batch_size, self.n_head, self.d_model, self.state_size))
        ys = []
        for t in range(0, seq_len):
            h = dA[:, :, t] * h + BX[:, :, t] # B, H, D, N
            y = paddle.einsum('bhdn,bhn->bhd', h, C[:, :, t])
            y = y.reshape([batch_size, -1])
            ys.append(y)
            
        y = paddle.stack(ys, axis=1) # B, L, D, N
        # hs = paddle.multiply(dA, h) + BX

        y = self.fc4(y) + D * x_in
        return y
