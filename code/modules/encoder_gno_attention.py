from torch import nn
from .supernode_pooling_gno import SupernodePooling
from .self_attn_patch import PatchifiedSelfAttentionBlocks


class EncoderGNOAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_dim,
            enc_depth,
            enc_num_heads,
            radius,
            patchify=True,
            P=2,
            H=None,
            W=None,
            init_weights="truncnormal",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.radius = radius
        assert patchify, "patchify must be True for this implementation"
        if patchify:
            assert H is not None and W is not None, \
                f"H and W must be provided if patchify is True"

        self.patchify = patchify
        self.P = P
        self.H = H
        self.W = W
        self.init_weights = init_weights

        # supernode pooling, use nonlinear kernel GNO (modified from GINO code) 
        self.supernode_pooling = SupernodePooling(
            radius=radius,
            input_dim=input_dim,
            hidden_dim=enc_dim,
            ndim=2,
        ) 
        
        self.patchified_self_attention_blocks = PatchifiedSelfAttentionBlocks(
            P=P,
            H=H,
            W=W,
            input_dim=enc_dim,
            dim=enc_dim,
            num_heads=enc_num_heads,
            enc_depth=enc_depth,
            init_weights=init_weights
        )

    def forward(self, input_feat, input_pos, query_pos):
        '''
        input_feat: (batch, ngp, channel)
        input_pos: (batch, ngp, 2)
        query_pos: (batch, ngp_latent, 2), inquired position for GNO
        '''

        # supernode pooling
        x = self.supernode_pooling(
            input_feat=input_feat,
            input_pos=input_pos,
            query_pos=query_pos
        )

        x = self.patchified_self_attention_blocks(x)  

        return x
