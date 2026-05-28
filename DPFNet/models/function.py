import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def drop_path(inputs, drop_prob=0.0, training=False):
    """drop path op
    Args:
        input: tensor with arbitrary shape
        drop_prob: float number of drop path probability, default: 0.0
        training: bool, if current mode is training, default: False
    Returns:
        output: output tensor after drop path
    """

    if drop_prob == 0.0 or not training:
        return inputs
    keep_prob = 1 - drop_prob
    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1) 
    random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
    random_tensor = random_tensor.floor()  
    output = (
        inputs.divide(keep_prob) * random_tensor
    ) 
    return output


class DropPath(nn.Layer):
    """DropPath class"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        return drop_path(inputs, self.drop_prob, self.training)

class Mlp(nn.Layer):
    """MLP module

    MLP using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc1 -> act -> dropout -> fc2 -> dropout

    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


class PatchEmbed(nn.Layer):
    """2D Image to Patch Embedding

    Apply patch embeddings on input images. Embeddings is implemented using a Conv2D op.

    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose((0, 2, 1))  # BCHW -> BNC
        x = self.norm(x)
        return x


class Identity(nn.Layer):
    """Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input