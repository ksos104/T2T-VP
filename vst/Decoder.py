from lib2to3.pgen2 import token
import einops
import torch.nn as nn
import torch
from .token_performer import Token_performer
from .Transformer import recon_token_inference, pred_token_inference, token_TransformerEncoder


class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.recon_token_pre = recon_token_inference(dim=embed_dim, num_heads=1)
        self.pred_token_pre = pred_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, recon_tokens, pred_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((recon_tokens, fea), dim=1)
        fea = torch.cat((fea, pred_tokens), dim=1)
        # [B, 1 + H*W + 1, 384]

        fea = self.encoderlayer(fea)
        # fea [B, 1 + H*W + 1, 384]
        recon_tokens = fea[:, 0, :].unsqueeze(1)
        pred_tokens = fea[:, -1, :].unsqueeze(1)

        recon_fea = self.recon_token_pre(fea)
        # recon_fea [B, H*W, 384]
        pred_fea = self.pred_token_pre(fea)
        # pred_fea [B, H*W, 384]

        # reproject back to 64 dim
        recon_fea = self.mlp2(self.norm2(recon_fea))
        pred_fea = self.mlp2_c(self.norm2_c(pred_fea))

        return recon_fea, pred_fea, fea, recon_tokens, pred_tokens


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        # if self.fuse:
        #     # [2] fuse encoder fea and decoder fea
        #     dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
        #     dec_fea = self.att(dec_fea)

        return dec_fea


class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_c = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        self.n_frames = 10
        # predict recon maps
        self.pre_1_16 = nn.Linear(token_dim*self.n_frames, self.n_frames)
        self.pre_1_8 = nn.Linear(token_dim*self.n_frames, self.n_frames)
        self.pre_1_4 = nn.Linear(token_dim*self.n_frames, self.n_frames)
        self.pre_1_1 = nn.Linear(token_dim*self.n_frames, self.n_frames)
        # predict pred maps
        self.pre_1_16_c = nn.Linear(token_dim*self.n_frames, self.n_frames)
        self.pre_1_8_c = nn.Linear(token_dim*self.n_frames, self.n_frames)
        self.pre_1_4_c = nn.Linear(token_dim*self.n_frames, self.n_frames)
        self.pre_1_1_c = nn.Linear(token_dim*self.n_frames, self.n_frames)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, recon_fea_1_16, token_fea_1_16, recon_tokens, pred_fea_1_16, pred_tokens, rgb_fea_1_8, rgb_fea_1_4):
        ## original
        # recon_fea_1_16 [B, 14*14, 384]
        # pred_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain recon token and pred token)

        # recon_tokens [B, 1, 384]
        # pred_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 56*56, 64]

        ## VP
        # recon_fea_1_16 [B*(4*4), 10, 384]
        # token_fea_1_16 [B*(4*4), 1 + 10 + 1, 384]
        # recon_tokens [B*(4*4), 1, 384]
        # pred_fea_1_16 [B*(4*4), 10, 384]
        # pred_tokens [B*(4*4), 1, 384]
        # rgb_fea_1_8 None
        # rgb_fea_1_4 None

        B, _, _, = token_fea_1_16.size()
        n_patch = (self.img_size // 16) ** 2

        recon_fea_1_16 = self.mlp(self.norm(recon_fea_1_16))
        # recon_fea_1_16 [B, 14*14, 64]
        recon_fea_1_16 = einops.rearrange(recon_fea_1_16, '(b n) t d -> b n (t d)', n=n_patch)
        recon_1_16 = self.pre_1_16(recon_fea_1_16)
        recon_1_16 = einops.rearrange(recon_1_16, 'b (nh nw) t -> b t nh nw', nh=self.img_size//16, nw=self.img_size//16)
        # recon_1_16 = recon_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        pred_fea_1_16 = self.mlp_c(self.norm_c(pred_fea_1_16))
        # pred_fea_1_16 [B, 14*14, 64]
        pred_fea_1_16 = einops.rearrange(pred_fea_1_16, '(b n) t d -> b n (t d)', n=n_patch)
        pred_1_16 = self.pre_1_16_c(pred_fea_1_16)
        pred_1_16 = einops.rearrange(pred_1_16, 'b (nh nw) t -> b t nh nw', nh=self.img_size//16, nw=self.img_size//16)
        # pred_1_16 = pred_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        token_fea_1_16 = einops.rearrange(token_fea_1_16[:, 1:-1, :], '(b n) t d -> (b t) n d', n=n_patch)
        fea_1_8 = self.decoder1(token_fea_1_16, rgb_fea_1_8)

        # token prediction
        fea_1_8 = einops.rearrange(fea_1_8, '(b t) n d -> (b n) t d', t=self.n_frames)
        recon_tokens = recon_tokens.expand(-1, 4, -1)
        recon_tokens = einops.rearrange(recon_tokens, 'bn c d -> (bn c) d').unsqueeze(dim=1)
        pred_tokens = pred_tokens.expand(-1, 4, -1)
        pred_tokens = einops.rearrange(pred_tokens, 'bn c d -> (bn c) d').unsqueeze(dim=1)
        recon_fea_1_8, pred_fea_1_8, token_fea_1_8, recon_tokens, pred_tokens = self.token_pre_1_8(fea_1_8, recon_tokens, pred_tokens)

        # predict recon maps and pred maps
        recon_fea_1_8 = einops.rearrange(recon_fea_1_8, '(b n) t d -> b n (t d)', n=n_patch*4)
        recon_1_8 = self.pre_1_8(recon_fea_1_8)
        recon_1_8 = einops.rearrange(recon_1_8, 'b (nh nw) t -> b t nh nw', nh=self.img_size//8, nw=self.img_size//8)
        # recon_1_8 = recon_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        pred_fea_1_8 = einops.rearrange(pred_fea_1_8, '(b n) t d -> b n (t d)', n=n_patch*4)
        pred_1_8 = self.pre_1_8_c(pred_fea_1_8)
        pred_1_8 = einops.rearrange(pred_1_8, 'b (nh nw) t -> b t nh nw', nh=self.img_size//8, nw=self.img_size//8)
        # pred_1_8 = pred_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        token_fea_1_8 = einops.rearrange(token_fea_1_8[:, 1:-1, :], '(b n) t d -> (b t) n d', n=n_patch*4)
        fea_1_4 = self.decoder2(token_fea_1_8, rgb_fea_1_4)

        # token prediction
        fea_1_4 = einops.rearrange(fea_1_4, '(b t) n d -> (b n) t d', t=self.n_frames)
        recon_tokens = recon_tokens.expand(-1, 4, -1)
        recon_tokens = einops.rearrange(recon_tokens, 'bn c d -> (bn c) d').unsqueeze(dim=1)
        pred_tokens = pred_tokens.expand(-1, 4, -1)
        pred_tokens = einops.rearrange(pred_tokens, 'bn c d -> (bn c) d').unsqueeze(dim=1)
        recon_fea_1_4, pred_fea_1_4, token_fea_1_4, recon_tokens, pred_tokens = self.token_pre_1_4(fea_1_4, recon_tokens, pred_tokens)

        # predict recon maps and pred maps
        recon_fea_1_4 = einops.rearrange(recon_fea_1_4, '(b n) t d -> b n (t d)', n=n_patch*16)
        recon_1_4 = self.pre_1_4(recon_fea_1_4)
        recon_1_4 = einops.rearrange(recon_1_4, 'b (nh nw) t -> b t nh nw', nh=self.img_size//4, nw=self.img_size//4)
        # recon_1_4 = recon_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        pred_fea_1_4 = einops.rearrange(pred_fea_1_4, '(b n) t d -> b n (t d)', n=n_patch*16)
        pred_1_4 = self.pre_1_4_c(pred_fea_1_4)
        pred_1_4 = einops.rearrange(pred_1_4, 'b (nh nw) t -> b t nh nw', nh=self.img_size//4, nw=self.img_size//4)
        # pred_1_4 = pred_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        recon_fea_1_4 = einops.rearrange(recon_fea_1_4, 'b n (t d) -> (b t) n d', t=self.n_frames)
        pred_fea_1_4 = einops.rearrange(pred_fea_1_4, 'b n (t d) -> (b t) n d', t=self.n_frames)
        recon_fea_1_1 = self.decoder3(recon_fea_1_4)
        pred_fea_1_1 = self.decoder3_c(pred_fea_1_4)

        recon_fea_1_1 = einops.rearrange(recon_fea_1_1, '(b t) n d -> b n (t d)', t=self.n_frames)
        recon_1_1 = self.pre_1_1(recon_fea_1_1)
        recon_1_1 = einops.rearrange(recon_1_1, 'b (nh nw) t -> b t nh nw', nh=self.img_size//1, nw=self.img_size//1)
        # recon_1_1 = recon_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        pred_fea_1_1 = einops.rearrange(pred_fea_1_1, '(b t) n d -> b n (t d)', t=self.n_frames)
        pred_1_1 = self.pre_1_1_c(pred_fea_1_1)
        pred_1_1 = einops.rearrange(pred_1_1, 'b (nh nw) t -> b t nh nw', nh=self.img_size//1, nw=self.img_size//1)
        # pred_1_1 = pred_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        return [recon_1_16, recon_1_8, recon_1_4, recon_1_1], [pred_1_16, pred_1_8, pred_1_4, pred_1_1]

