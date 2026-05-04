"""
From: https://github.com/yaohungt/Multimodal-Transformer
Paper: Multimodal Transformer for Unaligned Multimodal Language Sequences
"""
import os
import sys
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from models.subNets.transformers_encoder.transformer import TransformerEncoder

__all__ = ['MULT']


class MULT(nn.Module):
    def __init__(self, args):
        super(MULT, self).__init__()
        self.args = args
        self.use_bert = args.use_bert

        # Mult Model Initialization.
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # Personal feature fusion mode
        # decision / feature_gate / feature_gate_dim / feature_concat / none
        self.pf_fusion = getattr(args, "pf_fusion", "decision")
        pf_in = 9
        pf_hid = getattr(args, "personal_hidden", 32)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # --- PF modules (for feature-level or decision-level) ---
        # Feature-level: map pf -> combined_dim residual
        self.pf_to_h = nn.Sequential(
            nn.Linear(pf_in, pf_hid),
            nn.ReLU(),
            nn.Dropout(self.output_dropout),
            nn.Linear(pf_hid, combined_dim),
        )

        # Sample-wise gate: pf -> scalar in (0,1)
        self.pf_gate = nn.Sequential(
            nn.Linear(pf_in, pf_hid),
            nn.ReLU(),
            nn.Linear(pf_hid, 1),
            nn.Sigmoid()
        )

        # Dimension-wise gate: pf -> (B, combined_dim) in (0,1)
        self.pf_gate_dim = nn.Sequential(
            nn.Linear(pf_in, pf_hid),
            nn.ReLU(),
            nn.Linear(pf_hid, combined_dim),
            nn.Sigmoid()
        )

        # Feature-level concat: pf -> pf_hid then concat
        self.pf_embed = nn.Sequential(
            nn.Linear(pf_in, pf_hid),
            nn.ReLU(),
            nn.Dropout(self.output_dropout),
        )

        # Projection layers
        # If using concat, proj1 input changes
        if self.pf_fusion == "feature_concat":
            self.proj1 = nn.Linear(combined_dim + pf_hid, combined_dim)
        else:
            self.proj1 = nn.Linear(combined_dim, combined_dim)

        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # Decision-level head (your original)
        self.personal_head = nn.Sequential(
            nn.Linear(pf_in, pf_hid),
            nn.ReLU(),
            nn.Dropout(self.output_dropout),
            nn.Linear(pf_hid, output_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))  # gate for decision-level fusion

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout_a
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    def forward(self, text, audio, video, personal_feature=None):
        """
        personal_feature: Tensor[B,9] (recommended already normalized outside, e.g. fold-wise z-score)
        """
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]  # (N, 2*d_l)

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]  # (N, 2*d_a)

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]  # (N, 2*d_v)

        # fused feature
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # (B, combined_dim)

        # ---------------- Feature-level fusion (optional) ----------------
        if personal_feature is not None and self.pf_fusion in ["feature_gate", "feature_gate_dim", "feature_concat"]:
            if self.pf_fusion == "feature_gate":
                # sample-wise gated additive injection: h' = h + gate(pf) * delta(pf)
                delta = self.pf_to_h(personal_feature)  # (B, D)
                gate = self.pf_gate(personal_feature)   # (B, 1)
                last_hs = last_hs + gate * delta        # (B, D)

                # residual block
                last_hs_proj = self.proj2(
                    F.dropout(F.relu(self.proj1(last_hs), inplace=True),
                              p=self.output_dropout, training=self.training)
                )
                last_hs_proj = last_hs_proj + last_hs

            elif self.pf_fusion == "feature_gate_dim":
                # dimension-wise gated additive injection: h' = h + gate_dim(pf) ⊙ delta(pf)
                delta = self.pf_to_h(personal_feature)      # (B, D)
                gate_d = self.pf_gate_dim(personal_feature) # (B, D) in (0,1)
                contrib = gate_d * delta                    # (B, D)
                last_hs = last_hs + contrib

                # residual block
                last_hs_proj = self.proj2(
                    F.dropout(F.relu(self.proj1(last_hs), inplace=True),
                              p=self.output_dropout, training=self.training)
                )
                last_hs_proj = last_hs_proj + last_hs

            else:  # feature_concat
                pf_e = self.pf_embed(personal_feature)  # (B, pf_hid)
                h_cat = torch.cat([last_hs, pf_e], dim=1)  # (B, D+pf_hid)

                last_hs_proj = self.proj2(
                    F.dropout(F.relu(self.proj1(h_cat), inplace=True),
                              p=self.output_dropout, training=self.training)
                )
                # residual: add only the original last_hs (dimension matches combined_dim)
                last_hs_proj = last_hs_proj + last_hs

            logits_mult = self.out_layer(last_hs_proj)
            output = logits_mult  # already fused at feature-level

        else:
            # ---------------- Original MULT residual block ----------------
            last_hs_proj = self.proj2(
                F.dropout(F.relu(self.proj1(last_hs), inplace=True),
                          p=self.output_dropout, training=self.training)
            )
            last_hs_proj = last_hs_proj + last_hs
            logits_mult = self.out_layer(last_hs_proj)

            # ---------------- Decision-level fusion (your original) ----------------
            if personal_feature is not None and self.pf_fusion == "decision":
                logits_p = self.personal_head(personal_feature)  # (B, output_dim)
                a = torch.sigmoid(self.alpha)                   # scalar in (0,1)
                output = (1 - a) * logits_mult + a * logits_p
            else:
                output = logits_mult

        res = {
            'Feature_t': last_h_l,
            'Feature_a': last_h_a,
            'Feature_v': last_h_v,
            'Feature_f': last_hs,
            'M': output
        }

        # Only keep PF_gate_dim for gate-only importance (PF9_imp_gate_vec).
        # Do NOT return PF_delta / PF_contrib anymore.
        if (personal_feature is not None) and (self.pf_fusion == "feature_gate_dim"):
            res.update({
                'PF_gate_dim': gate_d
            })

        return res
