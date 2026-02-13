import torch.nn as nn

from .encoder_gno_attention import EncoderGNOAttention
from .decoder_attention import DecoderAttention

# Modified from https://github.com/yzshi5/MINO
class RENO(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        # encoder
        self.encoder = EncoderGNOAttention(
            **encoder_config,
            )

        decoder_input_dim = encoder_config['enc_dim']

        # decoder
        self.decoder = DecoderAttention(
            input_dim=decoder_input_dim,
            **decoder_config,
        )
        

    def forward(self, input_feat, input_pos, query_pos, src_pos, rec_pos):
        '''
        input_feat: (batch, ngp, nc), all features but source location
        input_pos: (batch, ngp, 2)
        query_pos: (batch, ngp_latent, 2), inquired position for GNO
        src_pos: (batch, nrec, 2)
        rec_pos: (batch, nrec, 2)
        #output_feat: (batch, nrec, channel), all features but source location
        '''
        assert src_pos.shape == rec_pos.shape, "src_pos and rec_pos must have the same shape"
        x = self.encoder(input_feat, input_pos, query_pos)  # (b, ngp_latent, nc)
        x = self.decoder(x, src_pos, rec_pos)  # (b, nrec, channel)
        return x
