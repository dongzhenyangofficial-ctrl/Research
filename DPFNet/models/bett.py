import math
from functools import partial
from models.mamba import Mamba
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.function import DropPath, Mlp, Identity
import numpy as np
from clip import tokenize, load_model

clip, transforms = load_model('ViT_B_32', pretrained=True)

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)

class CrossAttention(nn.Layer):
    """Cross Attention Layer"""

    def __init__(
        self,
        dim,
        num_heads=8,
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
        self.proj = nn.Linear(all_head_dim, dim)
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

class Attention(nn.Layer):
    """Attention Layer"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)
        if qkv_bias:

            self.q_bias = paddle.create_parameter(
                shape=[all_head_dim], dtype="float32", default_initializer=zeros_
            )

            self.v_bias = paddle.create_parameter(
                shape=[all_head_dim], dtype="float32", default_initializer=zeros_
            )
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3

            self.relative_position_bias_table = paddle.create_parameter(
                shape=[self.num_relative_distance, num_heads],
                dtype="float32",
                default_initializer=zeros_,
            )  # 2*Wh-1 * 2*Ww-1, nH
            
            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten.unsqueeze(
                axis=2
            ) - coords_flatten.unsqueeze(
                axis=1
            )  
            relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = paddle.zeros(
                [
                    window_size[0] * window_size[1] + 1,
                    window_size[0] * window_size[1] + 1,
                ],
                dtype=relative_coords.dtype,
            )
            # Wh*Ww, Wh*Ww
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat(
                (self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias)
            )

        qkv = F.linear(x=x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose([0, 1, 3, 2])

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape([-1])
            ].reshape(
                [
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1,
                    -1,
                ]
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1]
            )  # nH, Wh*Ww, Wh*Ww

            attn = attn + relative_position_bias.unsqueeze(axis=0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        self.norm2 = norm_layer(dim)
        self.attn2 = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values:

            self.gamma_1 = paddle.create_parameter(
                shape=[dim],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=init_values),
            )

            self.gamma_2 = paddle.create_parameter(
                shape=[dim],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=init_values),
            )

            self.gamma_3 = paddle.create_parameter(
                shape=[dim],
                dtype="float32",
                default_initializer=nn.initializer.Constant(value=init_values),
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, y):

        x = x + self.drop_path(
            self.gamma_1 * self.attn1(self.norm1(x), None)
        )
        x = x + self.drop_path(
            self.gamma_2 * self.attn2(self.norm2(x), self.norm2(y))
        )
        x = x + self.drop_path(self.gamma_3 * self.mlp(self.norm3(x)))

        return x


class RelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3

        self.relative_position_bias_table = paddle.create_parameter(
            shape=[self.num_relative_distance, num_heads],
            dtype="float32",
            default_initializer=zeros_,
        )  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten.unsqueeze(axis=2) - coords_flatten.unsqueeze(
            axis=1
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = paddle.zeros(
            [window_size[0] * window_size[1] + 1, window_size[0] * window_size[1] + 1]
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

def apply_gate(outputs, gates):
    gates = F.gelu(gates, approximate=True)
    outputs = gates * outputs
    return outputs

class Bett(nn.Layer):
    """Bett Layer"""

    def __init__(
        self,
        img_size=224,
        num_patches=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        init_scale=0.001
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        depth = depth // 3
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        # )
        # num_patches = self.patch_embed.num_patches

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype="float32",
            default_initializer=trunc_normal_,
        )

        self.mask_token  = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype="float32",
            default_initializer=trunc_normal_,
        )

        if use_abs_pos_emb:

            self.pos_embed = paddle.create_parameter(
                shape=[1, num_patches, embed_dim],
                dtype="float32",
                default_initializer=trunc_normal_,
            )
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.grid_size, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.mamba = Mamba(
            depth=depth,
            d_inner=512,
            d_model=64,
            state_size=16,
        )
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.grid_size
                    if use_rel_pos_bias
                    else None,
                )
                for i in range(depth)
            ]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.w = paddle.create_parameter(
            shape=[1], dtype="float32", default_initializer=ones_
        )
        self.temp = paddle.create_parameter(
            shape=[1], dtype="float32", default_initializer=nn.initializer.Constant(value=0.01)
        )
        
        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else Identity()
        self.attn1 = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.attn2 = CrossAttention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight)
            self.head.weight.set_value(
                self.head.weight.multiply(paddle.to_tensor(init_scale))
            )
            self.head.bias.set_value(
                self.head.bias.multiply(paddle.to_tensor(init_scale))
            )

    def fix_init_weight(self):
        def rescale(param, layer_id):

            param.set_value(param.divide(paddle.to_tensor(math.sqrt(2.0 * layer_id))))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn1.proj.weight, layer_id + 1)
            rescale(layer.attn2.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x, target):
        batch_size, seq_len, nc, sample_size, _ = x.shape
        x = x.reshape([batch_size * seq_len, nc, sample_size, sample_size])
        
        with paddle.no_grad():
            visual = clip.encode_image(x)
        visual = visual.reshape([batch_size, seq_len, -1])

        batch_size, seq_len, C = visual.shape

        cls_tokens = self.cls_token.expand([batch_size, -1, -1])

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None


        emos = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]
        auxs  = ['animal', 'scene', 'object', 'nature', 'building', 'adults', 'children', 'man', 'woman', 'child']
        text = tokenize(emos)
        auxs = tokenize(auxs)

        with paddle.no_grad():
            textual = clip.encode_text(text)
            auxs = clip.encode_text(auxs)

        textual = textual.unsqueeze(0).expand([batch_size, -1, -1])
        auxs = auxs.unsqueeze(0).expand([batch_size, -1, -1])

        if self.pos_embed is not None:
            visual = visual + self.pos_embed

        m_visual = visual + self.attn2(self.norm(visual), self.norm(visual))
        n_visual = visual + self.attn2(self.norm(visual), self.norm(auxs))

        n_cls_tokens = paddle.concat([cls_tokens, textual], axis=1)
        m_cls_tokens = n_cls_tokens.clone()

        for blk in self.blocks:
            m_cls_tokens = blk(m_cls_tokens, m_visual)
            n_cls_tokens = blk(n_cls_tokens, n_visual)

        n_cls_tokens = apply_gate(self.proj(n_cls_tokens), self.gate(paddle.concat([n_cls_tokens, m_cls_tokens], axis=-1)))
        coff = self.w * paddle.norm(m_cls_tokens, p=2, axis=-1, keepdim=True) / paddle.norm(n_cls_tokens, p=2, axis=-1, keepdim=True)
        coff = paddle.where(coff<1, coff, paddle.ones(shape=coff.shape))
        cls_tokens = m_cls_tokens + coff * n_cls_tokens

        F = cls_tokens[:, 0]
        T = cls_tokens[:, 1:]
        F /= F.norm(axis=-1, keepdim=True)
        T /= T.norm(axis=-1, keepdim=True)
        output = paddle.einsum('bc,blc->bl', F, T) / self.temp
        return output, F, T, T

    def forward(self, inputs, bool_masked_pos):
        x = self.forward_features(inputs, bool_masked_pos)
        # x = self.head(x)
        return x