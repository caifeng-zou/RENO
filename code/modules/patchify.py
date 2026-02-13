import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, P, H, W):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.npatch_H = H // P
        self.npatch_W = W // P
    
    def forward(self, x):
        '''
        x: (batch, seq_len, input_dim)
        -> (batch, npatch, input_dim*(P**2))
        '''
        assert x.shape[1] == self.H * self.W, f"Number of nodes in the input tensor must be equal to H * W, but got {x.shape[1]}"
        bs = len(x)
        if self.P == 1:
            return x

        x = x.view(bs, self.H, self.W, -1)
        x = x.view(bs, self.npatch_H, self.P, self.npatch_W, self.P, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [batch, nH, nW, P, P, input_dim]
        x = x.view(bs, self.npatch_H * self.npatch_W, -1)  # [batch, npatch, input_dim*(P**2)]
        return x

class UnPatchify(nn.Module):
    def __init__(self, P, H, W):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.npatch_H = H // P
        self.npatch_W = W // P
    
    def forward(self, x):
        '''
        x: (batch, npatch, input_dim*(P**2))
        -> (batch, seq_len, input_dim)
        '''
        # reshape back to original shape
        if self.P == 1:
            return x
            
        bs = len(x)
        x = x.view(bs, self.npatch_H, self.npatch_W, self.P, self.P, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [batch, nH, P, nW, P, input_dim]
        x = x.view(bs, self.H * self.W, -1)
        return x