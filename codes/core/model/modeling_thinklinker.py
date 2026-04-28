import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class ThinkLinkerEncoder(nn.Module):
    def __init__(self, args):
        super(ThinkLinkerEncoder, self).__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.pretrained_model)

        self.image_cls_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.dv)
        self.image_tokens_fc = nn.Linear(self.args.model.input_image_hidden_dim, self.args.model.dv)

        # 加载第一阶段预训练好的编码器
        ckpt_path = self.args.first_stage_ckpt
        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    sd_source = ckpt["state_dict"]
                else:
                    sd_source = ckpt

                sd_target_template = self.clip.state_dict()
                target_keys = list(sd_target_template.keys())
                mapped = {}
                source_keys = list(sd_source.keys())
                suffix_to_source = {}
                for sk in source_keys:
                    suffix_to_source.setdefault(sk, sk)
                found_count = 0
                new_state_dict = {}
                for tk in target_keys:
                    candidate = None
                    if tk in sd_source:
                        candidate = tk
                    else:
                        for sk in source_keys:
                            if sk.endswith(tk):
                                candidate = sk
                                break
                    if candidate is not None:
                        new_state_dict[tk] = sd_source[candidate]
                        found_count += 1

                missing_keys, unexpected_keys = self.clip.load_state_dict(new_state_dict, strict=False)
                print(
                    f"[INFO] Load encoder weights from {ckpt_path}. matched_keys={found_count}, missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
                if len(missing_keys) > 0:
                    print("[INFO] example missing keys (up to 10):", missing_keys[:10])
            except Exception as e:
                print(f"[ERROR] 加载第一阶段 checkpoint 时出错: {e}")

        # 冻结编码器参数
        if getattr(self.args, "freeze_clip_in_second_stage", True):
            self.clip.requires_grad_(False)


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                pixel_values=None):
        text_embeds = None
        image_embeds = None
        text_seq_tokens = None
        image_patch_tokens = None

        # -------- 文本-only 分支 --------
        if (input_ids is not None) and (pixel_values is None):
            text_outputs = self.clip.text_model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                return_dict=True)
            text_seq_tokens = text_outputs.last_hidden_state
            try:
                text_embeds = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                text_embeds = text_seq_tokens[:, 0, :]

        # -------- 双模态 分支（text + image） --------
        elif (input_ids is not None) and (pixel_values is not None):
            clip_output = self.clip(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    pixel_values=pixel_values)

            text_embeds = clip_output.text_embeds  # 文本全局(batch_size, 512)
            image_embeds = clip_output.image_embeds  # 图像全局(batch_size, 512)

            text_seq_tokens = clip_output.text_model_output[0]
            image_patch_tokens = clip_output.vision_model_output[0]

        if image_embeds is not None:
            image_embeds = self.image_cls_fc(image_embeds)  # (bs, 512) -> (bs, dv)
        if image_patch_tokens is not None:
            image_patch_tokens = self.image_tokens_fc(image_patch_tokens)  # (bs, patch, 96)

        return text_embeds, image_embeds, text_seq_tokens, image_patch_tokens


class TextLFTModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        d_in = args.model.input_hidden_dim
        self.d_proj = args.model.tlft_hidden_dim
        self.rank = args.model.rank_num

        self.fc = nn.Linear(d_in, self.d_proj)
        self.cls_factor = nn.Parameter(torch.empty(self.rank, self.d_proj + 1, self.d_proj))
        self.loc_factor = nn.Parameter(torch.empty(self.rank, self.d_proj + 1, self.d_proj))
        self.fusion_weights = nn.Parameter(torch.empty(self.rank))
        self.fusion_bias = nn.Parameter(torch.zeros(self.d_proj))

        self.res_mlp = nn.Linear(2 * self.d_proj, self.d_proj)
        self.layer_norm = nn.LayerNorm(self.d_proj)

        self._reset_parameters()

        d_proj = self.d_proj
        self.gate_layer = nn.Linear(3 * d_proj, 3)
        self.input_dropout = nn.Dropout(0.1)
        hidden_dim = 2 * d_proj
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_proj, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_proj)
        )
        self.layernorm = nn.LayerNorm(d_proj)

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        nn.init.xavier_normal_(self.cls_factor)
        nn.init.xavier_normal_(self.loc_factor)
        nn.init.normal_(self.fusion_weights, std=0.02)
        nn.init.zeros_(self.fusion_bias)

        nn.init.xavier_normal_(self.res_mlp.weight)
        nn.init.zeros_(self.res_mlp.bias)

    def _pool_attn(self, global_proj, local_seq):
        attn_scores = torch.matmul(global_proj.unsqueeze(2), local_seq.transpose(-1, -2))
        attn_probs = F.softmax(attn_scores / (self.d_proj ** 0.5), dim=-1)
        pooled = torch.matmul(attn_probs, local_seq).squeeze(2)
        return pooled

    def _hybrid_pool_and_gate(self, global_cls, local_feats):
        max_feats, _ = torch.max(local_feats, dim=2)
        mean_feats = torch.mean(local_feats, dim=2)
        attn_feats = self._pool_attn(global_cls, local_feats)

        pre_concat = torch.cat([max_feats, mean_feats, attn_feats], dim=-1)  # (bs, N, 3*d_proj)
        parts = torch.stack([max_feats, mean_feats, attn_feats], dim=2)  # parts: (bs, N, 3, d)

        gate_logits = self.gate_layer(pre_concat)
        gates = torch.sigmoid(gate_logits).unsqueeze(-1)

        gated_parts = parts * gates
        gated_concat = gated_parts.view(gated_parts.size(0), gated_parts.size(1), -1)

        gated_concat = self.input_dropout(gated_concat)
        fused = self.mlp(gated_concat)
        fused_feats = self.layernorm(fused)
        return fused_feats

    def _lowrank_map(self, v, factor):
        B, M, d = v.shape
        ones = v.new_ones(B, M, 1)
        v_w1 = torch.cat([ones, v], dim=-1)
        out = torch.einsum("bmd, rdf -> brmf", v_w1, factor)
        return out

    def _fuse_single(self, cls_proj, loc_proj, cls_factor, loc_factor, fusion_weights, fusion_bias):
        cls_out = self._lowrank_map(cls_proj, cls_factor)
        loc_out = self._lowrank_map(loc_proj, loc_factor)
        fused_ranked = cls_out * loc_out
        fw = fusion_weights.view(1, -1, 1, 1)
        weighted = fused_ranked * fw
        fused = weighted.sum(dim=1) + fusion_bias
        fused = F.relu(fused)
        return fused

    def _build_context(self, global_vec, local_tokens):
        glob_proj = self.fc(global_vec)  # (B, N, d_proj)
        loc_proj = self.fc(local_tokens)  # (B, N, L, d_proj)

        loc_ctx = self._hybrid_pool_and_gate(glob_proj, loc_proj)

        fused = self._fuse_single(glob_proj, loc_ctx,
                                  self.cls_factor, self.loc_factor,
                                  self.fusion_weights, self.fusion_bias)  # (B,N,d)

        res = F.relu(self.res_mlp(torch.cat([glob_proj, loc_ctx], dim=-1)))  # (B,N,d)
        ctx = self.layer_norm(fused + res)  # (B,N,d)
        return ctx

    def forward(self, entity_text_cls, entity_text_tokens,mention_text_cls, mention_text_tokens):

        mention_text_cls = mention_text_cls.unsqueeze(1) # (B,1,d)
        mention_text_tokens = mention_text_tokens.unsqueeze(1) # (B,1,L,d)
        ment_ctx = self._build_context(mention_text_cls, mention_text_tokens)  # (B,1,d)

        ent_ctx = self._build_context(entity_text_cls, entity_text_tokens)     # (B,N,d)

        scores = torch.matmul(ment_ctx, ent_ctx.transpose(-1, -2)).squeeze(1)  # (B,N)
        return scores

class VisualLFTModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        d_in = args.model.dv
        self.d_proj = args.model.vlft_hidden_dim
        self.rank = args.model.rank_num

        self.fc = nn.Linear(d_in, self.d_proj)
        self.cls_factor = nn.Parameter(torch.empty(self.rank, self.d_proj + 1, self.d_proj))
        self.loc_factor = nn.Parameter(torch.empty(self.rank, self.d_proj + 1, self.d_proj))
        self.fusion_weights = nn.Parameter(torch.empty(self.rank))
        self.fusion_bias = nn.Parameter(torch.zeros(self.d_proj))

        self.res_mlp = nn.Linear(2 * self.d_proj, self.d_proj)
        self.layer_norm = nn.LayerNorm(self.d_proj)

        self._reset_parameters()

        d_proj = self.d_proj
        self.gate_layer = nn.Linear(3 * d_proj, 3)
        self.input_dropout = nn.Dropout(0.1)
        hidden_dim = 2 * d_proj
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_proj, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_proj)
        )
        self.layernorm = nn.LayerNorm(d_proj)

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        nn.init.xavier_normal_(self.cls_factor)
        nn.init.xavier_normal_(self.loc_factor)
        nn.init.normal_(self.fusion_weights, std=0.02)
        nn.init.zeros_(self.fusion_bias)

        nn.init.xavier_normal_(self.res_mlp.weight)
        nn.init.zeros_(self.res_mlp.bias)

    def _pool_attn(self, global_proj, local_seq):
        attn_scores = torch.matmul(global_proj.unsqueeze(2), local_seq.transpose(-1, -2))
        attn_probs = F.softmax(attn_scores / (self.d_proj ** 0.5), dim=-1)
        pooled = torch.matmul(attn_probs, local_seq).squeeze(2)
        return pooled

    def _hybrid_pool_and_gate(self, global_cls, local_feats):
        max_feats, _ = torch.max(local_feats, dim=2)  # (bs, N, d_proj)
        mean_feats = torch.mean(local_feats, dim=2)  # (bs, N, d_proj)
        attn_feats = self._pool_attn(global_cls, local_feats)  # (bs, N, d_proj)

        pre_concat = torch.cat([max_feats, mean_feats, attn_feats], dim=-1)  # (bs, N, 3*d_proj)
        parts = torch.stack([max_feats, mean_feats, attn_feats], dim=2)

        gate_logits = self.gate_layer(pre_concat)
        gates = torch.sigmoid(gate_logits).unsqueeze(-1)

        gated_parts = parts * gates
        gated_concat = gated_parts.view(gated_parts.size(0), gated_parts.size(1), -1)

        gated_concat = self.input_dropout(gated_concat)
        fused = self.mlp(gated_concat)
        fused_feats = self.layernorm(fused)
        return fused_feats

    def _lowrank_map(self, v, factor):
        B, M, d = v.shape
        ones = v.new_ones(B, M, 1)
        v_w1 = torch.cat([ones, v], dim=-1)
        out = torch.einsum("bmd, rdf -> brmf", v_w1, factor)
        return out

    def _fuse_single(self, cls_proj, loc_proj, cls_factor, loc_factor, fusion_weights, fusion_bias):
        cls_out = self._lowrank_map(cls_proj, cls_factor)
        loc_out = self._lowrank_map(loc_proj, loc_factor)
        fused_ranked = cls_out * loc_out
        fw = fusion_weights.view(1, -1, 1, 1)
        weighted = fused_ranked * fw
        fused = weighted.sum(dim=1) + fusion_bias
        fused = F.relu(fused)
        return fused

    def _build_context(self, global_vec, local_tokens):
        glob_proj = self.fc(global_vec)
        loc_proj = self.fc(local_tokens)

        loc_ctx = self._hybrid_pool_and_gate(glob_proj, loc_proj)

        fused = self._fuse_single(glob_proj, loc_ctx,
                                  self.cls_factor, self.loc_factor,
                                  self.fusion_weights, self.fusion_bias)

        res = F.relu(self.res_mlp(torch.cat([glob_proj, loc_ctx], dim=-1)))
        ctx = self.layer_norm(fused + res)
        return ctx

    def forward(self, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens):

        mention_image_cls = mention_image_cls.unsqueeze(1)  # (B,1,d)
        mention_text_tokens = mention_image_tokens.unsqueeze(1)  # (B,1,L,d)
        ment_ctx = self._build_context(mention_image_cls, mention_text_tokens)  # (B,1,d)

        ent_ctx = self._build_context(entity_image_cls, entity_image_tokens)
        scores = torch.matmul(ment_ctx, ent_ctx.transpose(-1, -2)).squeeze(1)
        return scores


class CrossLFTModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        d_text_in = args.model.input_hidden_dim
        d_img_in = args.model.dv

        d_proj = args.model.clft_hidden_dim
        self.d_proj = d_proj
        self.rank = args.model.rank_num

        self.text_fc = nn.Linear(d_text_in, d_proj)
        self.image_fc = nn.Linear(d_img_in, d_proj)

        self.gate_fc = nn.Linear(d_proj, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(d_proj)
        self.context_layer_norm = nn.LayerNorm(d_proj)

        self.text_cls_factor = nn.Parameter(torch.empty(self.rank, d_proj + 1, d_proj))
        self.text_loc_factor = nn.Parameter(torch.empty(self.rank, d_proj + 1, d_proj))
        self.img_cls_factor = nn.Parameter(torch.empty(self.rank, d_proj + 1, d_proj))
        self.img_loc_factor = nn.Parameter(torch.empty(self.rank, d_proj + 1, d_proj))

        self.fusion_weights = nn.Parameter(torch.empty(self.rank))
        self.fusion_bias = nn.Parameter(torch.zeros(d_proj))

        self.res_mlp = nn.Linear(2 * d_proj, d_proj)
        nn.init.xavier_normal_(self.res_mlp.weight)
        if self.res_mlp.bias is not None:
            nn.init.zeros_(self.res_mlp.bias)
        self._reset_parameters()

        self.gate_layer = nn.Linear(3 * d_proj, 3)
        self.input_dropout = nn.Dropout(0.1)
        hidden_dim = 2 * d_proj
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_proj, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_proj)
        )
        self.layernorm = nn.LayerNorm(d_proj)

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.text_fc.weight)
        if self.text_fc.bias is not None:
            nn.init.zeros_(self.text_fc.bias)

        nn.init.xavier_normal_(self.image_fc.weight)
        if self.image_fc.bias is not None:
            nn.init.zeros_(self.image_fc.bias)

        nn.init.xavier_normal_(self.text_cls_factor)
        nn.init.xavier_normal_(self.text_loc_factor)
        nn.init.xavier_normal_(self.img_cls_factor)
        nn.init.xavier_normal_(self.img_loc_factor)

        nn.init.normal_(self.fusion_weights, std=0.02)

    def _pool_attn(self, global_proj, local_seq):
        attn_scores = torch.matmul(global_proj.unsqueeze(2), local_seq.transpose(-1, -2))
        attn_probs = F.softmax(attn_scores / (self.d_proj ** 0.5), dim=-1)
        pooled = torch.matmul(attn_probs, local_seq).squeeze(2)
        return pooled

    def _hybrid_pool_and_gate(self, global_cls, local_feats):
        max_feats, _ = torch.max(local_feats, dim=2)  # (bs, N, d_proj)
        mean_feats = torch.mean(local_feats, dim=2)  # (bs, N, d_proj)
        attn_feats = self._pool_attn(global_cls, local_feats)  # (bs, N, d_proj)

        pre_concat = torch.cat([max_feats, mean_feats, attn_feats], dim=-1)  # (bs, N, 3*d_proj)
        parts = torch.stack([max_feats, mean_feats, attn_feats], dim=2)  # parts: (bs, N, 3, d)

        gate_logits = self.gate_layer(pre_concat)
        gates = torch.sigmoid(gate_logits).unsqueeze(-1)

        gated_parts = parts * gates
        gated_concat = gated_parts.view(gated_parts.size(0), gated_parts.size(1), -1)  # gated_concat: (bs, N, 3*d)

        gated_concat = self.input_dropout(gated_concat)

        fused = self.mlp(gated_concat)
        fused_feats = self.layernorm(fused)

        return fused_feats

    def _lowrank_map(self, v, factor):
        bs, N, d = v.shape
        ones = torch.ones(bs, N, 1, dtype=v.dtype, device=v.device)  # (bs, N, 1)
        v_w1 = torch.cat([ones, v], dim=-1)  # (bs, N, d+1)
        out = torch.einsum("bnd, rdf -> brnf", v_w1, factor)  # (bs, rank, N, d)
        return out

    def _fuse_modal(self, t_cls_proj, t_loc_proj, i_cls_proj, i_loc_proj,
                         t_cls_factor, t_loc_factor, i_cls_factor, i_loc_factor):
        t_cls_out = self._lowrank_map(t_cls_proj, t_cls_factor)
        t_loc_out = self._lowrank_map(t_loc_proj, t_loc_factor)
        i_cls_out = self._lowrank_map(i_cls_proj, i_cls_factor)
        i_loc_out = self._lowrank_map(i_loc_proj, i_loc_factor)

        fused_ranked = t_cls_out * t_loc_out * i_cls_out * i_loc_out  # (bs, rank, N, d_proj)

        fw = self.fusion_weights.view(1, self.rank, 1, 1)  # (1, rank, 1, 1)
        weighted = fused_ranked * fw  # (bs, rank, N, d)
        fused = weighted.sum(dim=1) + self.fusion_bias  # (bs, N, d)
        fused = F.relu(fused) # (bs, N, d)

        return fused

    def _build_context(self, text_cls, text_tokens, image_cls, image_tokens):
        t_cls_proj = self.text_fc(text_cls)
        i_cls_proj = self.image_fc(image_cls)
        t_tokens_proj = self.text_fc(text_tokens)
        i_tokens_proj = self.image_fc(image_tokens)

        t_loc_pooled = self._hybrid_pool_and_gate(t_cls_proj, t_tokens_proj)
        i_loc_pooled = self._hybrid_pool_and_gate(i_cls_proj, i_tokens_proj)

        fused = self._fuse_modal(
            t_cls_proj, t_loc_pooled, i_cls_proj, i_loc_pooled,
            self.text_cls_factor, self.text_loc_factor, self.img_cls_factor, self.img_loc_factor
        )  # (bs, N, d)

        concat_ti = torch.cat([t_cls_proj, i_loc_pooled], dim=-1)
        m = F.relu(self.res_mlp(concat_ti))
        u = fused + m
        context = self.context_layer_norm(u)
        return context

    def forward(self,
                entity_text_cls, entity_text_tokens,
                mention_text_cls, mention_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens,
                base_embedding=None):
        entity_context = self._build_context(entity_text_cls, entity_text_tokens,
                                             entity_image_cls, entity_image_tokens)

        mention_image_cls = mention_image_cls.unsqueeze(1)
        mention_image_tokens = mention_image_tokens.unsqueeze(1)
        mention_text_cls = mention_text_cls.unsqueeze(1)
        mention_text_tokens = mention_text_tokens.unsqueeze(1)
        mention_context = self._build_context(mention_text_cls, mention_text_tokens,
                                              mention_image_cls, mention_image_tokens)

        score = torch.matmul(mention_context, entity_context.transpose(-1, -2)).squeeze(1)
        return score


class ThinkLinkerMatcher(nn.Module):
    def __init__(self, args):
        super(ThinkLinkerMatcher, self).__init__()
        self.args = args
        self.t_lft = TextLFTModule(self.args)  # T-LTF
        self.v_lft = VisualLFTModule(self.args)  # V-LTF
        self.cross_lft = CrossLFTModule(self.args)  # Cross-LFT

        self.text_cls_layernorm = nn.LayerNorm(self.args.model.dt)
        self.text_tokens_layernorm = nn.LayerNorm(self.args.model.dt)
        self.image_cls_layernorm = nn.LayerNorm(self.args.model.dv)
        self.image_tokens_layernorm = nn.LayerNorm(self.args.model.dv)

    def forward(self,
                entity_text_cls, entity_text_tokens,
                mention_text_cls, mention_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens):
        entity_text_cls = self.text_cls_layernorm(entity_text_cls)
        mention_text_cls = self.text_cls_layernorm(mention_text_cls)

        entity_text_tokens = self.text_tokens_layernorm(entity_text_tokens)
        mention_text_tokens = self.text_tokens_layernorm(mention_text_tokens)

        entity_image_cls = self.image_cls_layernorm(entity_image_cls)
        mention_image_cls = self.image_cls_layernorm(mention_image_cls)

        entity_image_tokens = self.image_tokens_layernorm(entity_image_tokens)
        mention_image_tokens = self.image_tokens_layernorm(mention_image_tokens)


        text_matching_score = self.t_lft(entity_text_cls, entity_text_tokens,
                                        mention_text_cls, mention_text_tokens) # T-LTF
        image_matching_score = self.v_lft(entity_image_cls, entity_image_tokens,
                                         mention_image_cls, mention_image_tokens)  # V-LTF
        image_text_matching_score = self.cross_lft(entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens,
                                               entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens) # Cross-LFT

        score = (text_matching_score + image_matching_score + image_text_matching_score) / 3
        return score, (text_matching_score, image_matching_score, image_text_matching_score)