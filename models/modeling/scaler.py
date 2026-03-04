import math

import torch
import torch.nn.functional as F
from torch import nn



class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


class ConvScaler(nn.Module):
    def __init__(self, encoder_patch_size: "tuple[int, int]", embed_dim: int = 1024):
        super().__init__()
        self.encoder_patch_size = encoder_patch_size
        self.embed_dim = embed_dim

        max_patch_size = max(encoder_patch_size[0], encoder_patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        self.scale_blocks = nn.Sequential(
            *[ScaleBlock(embed_dim) for _ in range(num_upscale)]
        )

    def forward(self, x, **kwargs):
        return self.scale_blocks(x)


class ConvScalerFixedDepth(nn.Module):
    def __init__(self, depth: int, embed_dim: int = 1024):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        self.scale_blocks = nn.Sequential(
            *[ScaleBlock(embed_dim) for _ in range(depth)]
        )

    def forward(self, x, **kwargs):
        return self.scale_blocks(x)
