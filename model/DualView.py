import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, 1) for out_ch in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1)
        ])
        
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, 1)
        )
    
    def forward(self, feats):  # feats: list of [B, C, H, W]
        out = []
        for i, x in enumerate(feats):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        l1, l2, l3, l4 = out
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        
        path4 = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        path3 = self.scratch.refinenet3(path4, l3_rn, size=l2_rn.shape[2:])
        path2 = self.scratch.refinenet2(path3, l2_rn, size=l1_rn.shape[2:])
        path1 = self.scratch.refinenet1(path2, l1_rn)
        
        return self.scratch.output_conv(path1)


class DualViewDPT(nn.Module):
    def __init__(
        self,
        encoder_size='base',
        in_chns=1,
        seg_nclass=3,
        cls_nclass=1,
        features=256,
        use_bn=False,
    ):
        super().__init__()
        
        # Input adapter
        self.in_adapter = nn.Identity() if in_chns == 3 else nn.Conv2d(in_chns, 3, 1, bias=False)
        
        # Backbone
        self.backbone = DINOv2(model_name=encoder_size)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = 14
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }[encoder_size]
        
        # Decoder channels
        out_channels_map = {
            'small': [96, 192, 384, 384],
            'base': [96, 192, 384, 768],
            'large': [256, 512, 1024, 1024],
            'giant': [384, 768, 1536, 1536],
        }
        out_channels = out_channels_map[encoder_size]
        
        # Two independent decoders
        self.decoder_long = DPTHead(seg_nclass, self.embed_dim, features, use_bn, out_channels)
        self.decoder_trans = DPTHead(seg_nclass, self.embed_dim, features, use_bn, out_channels)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(128, cls_nclass)
        )
        
        # For advanced FP (CompDrop-style)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def _get_intermediate_features(self, x):
        """x: [B, 3, H, W] → returns list of [B, N, C]"""
        feats = self.backbone.get_intermediate_layers(x, self.intermediate_layer_idx)
        patch_h = x.shape[-2] // self.patch_size
        patch_w = x.shape[-1] // self.patch_size
        return feats, (patch_h, patch_w)

    def _token_to_2d(self, tokens, patch_h, patch_w):
        """Convert [B, N, C] → [B, C, H, W]"""
        return tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

    def _apply_comp_drop(self, feats_list):
        """Advanced channel-wise perturbation like your new DPT."""
        # Assume feats_list[0]: [B, C, H, W]
        B, C = feats_list[0].shape[:2]
        assert B % 2 == 0, "Batch size must be even for comp drop"
        
        # Sample mask for half batch
        mask1 = self.binomial.sample((B // 2, C)).to(feats_list[0].device) * 2.0  # scale to {0,2}
        mask2 = 2.0 - mask1
        
        # Optional: keep some channels unchanged (like original)
        dropout_prob = 0.5
        num_kept = int(B // 2 * (1 - dropout_prob))
        if num_kept > 0:
            kept = torch.randperm(B // 2)[:num_kept]
            mask1[kept] = 1.0
            mask2[kept] = 1.0
        
        mask = torch.cat([mask1, mask2], dim=0)  # [B, C]
        mask = mask.view(B, C, 1, 1)  # broadcast to [B, C, H, W]
        
        return [f * mask for f in feats_list]

    def forward(self, x_long, x_trans, need_fp=False):
        assert x_long.shape == x_trans.shape, "Inputs must have same shape"
        B, _, H, W = x_long.shape
        
        # Single forward through backbone
        x = torch.cat([x_long, x_trans], dim=0)  # [2B, C, H, W]
        x = self.in_adapter(x)
        token_feats, (ph, pw) = self._get_intermediate_features(x)  # each: [2B, N, C]
        
        # Split and reshape to 2D
        feats_long = [self._token_to_2d(f[:B], ph, pw) for f in token_feats]   # [B, C, H', W']
        feats_trans = [self._token_to_2d(f[B:], ph, pw) for f in token_feats]  # [B, C, H', W']
        
        if need_fp:
            # Apply CompDrop-style perturbation on 2D features
            p_feats_long = self._apply_comp_drop(feats_long)
            p_feats_trans = self._apply_comp_drop(feats_trans)
            
            # Decode clean
            seg_long_clean = self.decoder_long(feats_long)
            seg_trans_clean = self.decoder_trans(feats_trans)
            
            # Decode perturbed
            seg_long_fp = self.decoder_long(p_feats_long)
            seg_trans_fp = self.decoder_trans(p_feats_trans)
            
            # Upsample
            seg_long_clean = F.interpolate(seg_long_clean, (H, W), mode='bilinear', align_corners=True)
            seg_long_fp = F.interpolate(seg_long_fp, (H, W), mode='bilinear', align_corners=True)
            seg_trans_clean = F.interpolate(seg_trans_clean, (H, W), mode='bilinear', align_corners=True)
            seg_trans_fp = F.interpolate(seg_trans_fp, (H, W), mode='bilinear', align_corners=True)
            
            # Classification: use global mean of last token feature (before 2D reshape)
            feat_L = token_feats[-1][:B].mean(dim=1)      # [B, C]
            feat_T = token_feats[-1][B:].mean(dim=1)      # [B, C]
            feat_L_fp = self._token_to_2d(token_feats[-1][:B], ph, pw).mean(dim=[2,3])  # alternative
            feat_T_fp = self._token_to_2d(token_feats[-1][B:], ph, pw).mean(dim=[2,3])
            
            cls_clean = self.cls_head(torch.cat([feat_L, feat_T], dim=1))
            cls_fp = self.cls_head(torch.cat([feat_L_fp, feat_T_fp], dim=1))
            
            return (seg_long_clean, seg_long_fp), (seg_trans_clean, seg_trans_fp), (cls_clean, cls_fp)
        
        else:
            seg_long = self.decoder_long(feats_long)
            seg_trans = self.decoder_trans(feats_trans)
            seg_long = F.interpolate(seg_long, (H, W), mode='bilinear', align_corners=True)
            seg_trans = F.interpolate(seg_trans, (H, W), mode='bilinear', align_corners=True)
            
            feat_L = token_feats[-1][:B].mean(dim=1)
            feat_T = token_feats[-1][B:].mean(dim=1)
            cls_logits = self.cls_head(torch.cat([feat_L, feat_T], dim=1))
            
            return seg_long, seg_trans, cls_logits