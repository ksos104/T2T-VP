import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder

import einops


class VST(nn.Module):
    def __init__(self, args):
        super(VST, self).__init__()

        # VST Encoder
        self.backbone = T2t_vit_t_14(pretrained=False, args=args)

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input):
        ## image_Input.shape = [B, 10, 1, 64, 64]
        B, T, _, _, _ = image_Input.shape
        # VST Encoder
        fea_1_16, fea_1_8, fea_1_4 = self.backbone(image_Input)

        # VST Convertor
        fea_1_16 = einops.rearrange(fea_1_16, 'b t n d -> b (t n) d')
        fea_1_16 = self.transformer(fea_1_16)
        fea_1_16 = einops.rearrange(fea_1_16, 'b (t n) d -> b t n d', t=T)
        # fea_1_16 [B, 10, 4*4, 384]

        # VST Decoder
        fea_1_16 = einops.rearrange(fea_1_16, 'b t n d -> (b n) t d')
        recon_fea_1_16, fea_1_16, recon_tokens, pred_fea_1_16, pred_tokens = self.token_trans(fea_1_16)
        # recon_fea_1_16 [B*(4*4), 10, 384]
        # fea_1_16 [B*(4*4), 1 + 10 + 1, 384]
        # recon_tokens [B*(4*4), 1, 384]
        # pred_fea_1_16 [B*(4*4), 10, 384]
        # pred_tokens [B*(4*4), 1, 384]

        outputs = self.decoder(recon_fea_1_16, fea_1_16, recon_tokens, pred_fea_1_16, pred_tokens, fea_1_8, fea_1_4)

        return outputs
