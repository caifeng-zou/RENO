import torch

from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverBlock
from torch import nn
from .pos_emb import ContinuousSincosEmbed
from .self_attn_patch import PatchifiedSelfAttentionBlocks


class DecoderAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        #val_dim,
        output_dim,
        dec_dim,
        dec_depth,
        dec_num_heads,
        enforce_reciprocity=True,
        patchify=True,
        P=2,
        H=None,
        W=None,
        init_weights='truncnormal002',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enforce_reciprocity = enforce_reciprocity
        self.input_dim = input_dim
        #self.val_dim = val_dim
        self.output_dim = output_dim
        self.dec_dim = dec_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.init_weights = init_weights
        self.input_proj = LinearProjection(input_dim, dec_dim, init_weights=init_weights, optional=True)
        assert patchify, "patchify must be True for this implementation"
        if patchify:
            assert H is not None and W is not None, \
                f"H and W must be provided if patchify is True"

        self.blocks = nn.ModuleList(
            [
                SelfCrossAttentionBlock(
                    dim=dec_dim,
                    num_heads=dec_num_heads,
                    P=P,
                    H=H,
                    W=W,
                    init_weights=init_weights,
                )
                for _ in range(dec_depth)
            ]
        )

        self.pos_embed = ContinuousSincosEmbed(dim=dec_dim, ndim=2)

        # self.val_embed = nn.Sequential(
        #     LinearProjection(val_dim, dec_dim, init_weights=init_weights),
        #     nn.GELU(),
        #     LinearProjection(dec_dim, dec_dim, init_weights=init_weights),
        # )

        self.query_proj = nn.Sequential(
            LinearProjection(dec_dim*2, dec_dim*2, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(dec_dim*2, dec_dim, init_weights=init_weights),
        )

        # projection
        self.pred = nn.Sequential(
            # nn.LayerNorm(dec_dim, eps=1e-6),  
            LinearProjection(dec_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, src_pos, rec_pos):
        '''
        x: (batch, point, channel)
        src_pos: (batch, nrec, 2)
        rec_pos: (batch, nrec, 2)
        #output_val: (batch, point, channel)
        -> x: (batch, point, channel)
        '''
        x = self.input_proj(x)
        src_pos = self.pos_embed(src_pos)
        rec_pos = self.pos_embed(rec_pos)
        if self.enforce_reciprocity:
            query_pos1 = torch.cat([src_pos, rec_pos], dim=-1)
            query_pos2 = torch.cat([rec_pos, src_pos], dim=-1)
            query1 = self.query_proj(query_pos1)
            query2 = self.query_proj(query_pos2)
            query = (query1 + query2) / 2
        else:
            query_pos = torch.cat([src_pos, rec_pos], dim=-1)
            query = self.query_proj(query_pos)

        for block in self.blocks:
            x, query = block(x, query)

        query = self.pred(query)

        return query


class SelfCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, P, H, W, init_weights):
        super().__init__()

        self.self_attn = PatchifiedSelfAttentionBlocks(
            P=P,
            H=H,
            W=W,
            input_dim=dim,
            dim=dim,
            num_heads=num_heads,
            enc_depth=1,
            init_weights=init_weights
        )

        self.cross_attn = PerceiverBlock(
            kv_dim=dim, 
            dim=dim,
            num_heads=num_heads,
            init_weights=init_weights,
        )

    def forward(self, x, query):
        x = self.self_attn(x)
        query = self.cross_attn(q=query, kv=x)
        return x, query