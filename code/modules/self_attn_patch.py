import torch
import torch.nn as nn
from .pos_emb import ContinuousSincosEmbed
from .patchify import Patchify, UnPatchify
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverBlock


class PatchifiedSelfAttentionBlocks(nn.Module):
    def __init__(self, P, H, W, input_dim, dim, num_heads, enc_depth, init_weights):
        super().__init__()
        assert H % P == 0 and W % P == 0, \
            f"Dimensions must be divisible by patch size"
        self.P = P  # patch size
        self.H = H  # latent ny
        self.W = W  # latent nz
        assert H % P == 0 and W % P == 0, f"Dimensions must be divisible by patch size"
        self.npatch_H = H // P
        self.npatch_W = W // P
        self.input_dim = input_dim
        self.dim = dim
        self.num_heads = num_heads
        self.enc_depth = enc_depth
        self.init_weights = init_weights
        self.block = PerceiverBlock(
                dim=dim*(P**2), 
                num_heads=num_heads, 
                kv_dim=dim*(P**2), 
                init_weights=init_weights
                )
        self.blocks = nn.ModuleList(
            self.block for _ in range(enc_depth)
        )
        self.patchify = Patchify(P=P, H=H, W=W)
        self.unpatchify = UnPatchify(P=P, H=H, W=W)
        self.patch_pos = self._get_patch_pos()
        self.pos_emb = ContinuousSincosEmbed(dim=input_dim*(P**2), ndim=2)
        self.embbed_patch_pos = self.pos_emb(self.patch_pos)
        self.patch_linear = LinearProjection(2*input_dim*(P**2), dim*(P**2), init_weights=init_weights, optional=True)

    def forward(self, x):
        '''
        x: (batch, seq_len, input_dim)
        adapted from https://github.com/Shizheng-Wen/GAOT-3D/blob/main/src/model/gaot_3d.py#L86
        '''
        # patchify
        x = self.patchify(x)  # (batch, npatch, input_dim*(P**3))

        # positional embedding
        x = torch.cat([x, self.embbed_patch_pos.to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=-1)  # [batch, npatch, 2*input_dim*(P**2)]

        # linear projection
        x = self.patch_linear(x)  # [batch, npatch, dim*(P**2)]

        # self-attention blocks
        for blk in self.blocks:
            x = blk(q=x, kv=x)

        x = self.unpatchify(x)

        return x
    
    def _get_patch_pos(self):
        '''
        get the position of the patches
        '''
        pos = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, self.npatch_H, dtype=torch.float32),
                torch.linspace(0, 1, self.npatch_W, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 2)  # (npatch_H * npatch_W, 2)
        return pos