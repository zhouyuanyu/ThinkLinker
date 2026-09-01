import torch
import torch.nn as nn
import torch.nn.functional as F

class MentionEnhancer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.dim = args.model.dt
        self.attn_hidden = 128

        # ANW
        self.concat_mlp = nn.Sequential(
                nn.Linear(self.dim * 2, self.attn_hidden),
                nn.ReLU(),
                nn.Linear(self.attn_hidden, 1)
            )

        # RGA
        self.global_proj = nn.Linear(self.dim * 2, self.dim)
        self.local_proj = nn.Linear(self.dim * 2, self.dim)
        nn.init.xavier_uniform_(self.global_proj.weight)
        nn.init.xavier_uniform_(self.local_proj.weight)
        self.gate_global = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.Sigmoid()
        )
        self.gate_local = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.Sigmoid()
        )

    def compute_weights(self, orig_global, neigh_global):
        B, K, D = neigh_global.shape
        device = orig_global.device

        orig_expand = orig_global.unsqueeze(1).expand(-1, K, -1)  # [B,K,D]
        concat = torch.cat([orig_expand, neigh_global], dim=-1)  # [B,K,2D]
        scores = self.concat_mlp(concat.view(B * K, -1)).view(B, K)  # [B,K]
        weights = F.softmax(scores, dim=1)

        return weights  # [B,K]

    def aggregate_global(self, orig_global, neigh_global, weights):
        w = weights.unsqueeze(-1)  # [B,K,1]
        global_sum = torch.sum(w * neigh_global, dim=1)  # [B,D]
        return global_sum

    def aggregate_local(self, orig_local, neigh_local, weights):
        B, K, L, D = neigh_local.shape
        w = weights.unsqueeze(-1).unsqueeze(-1)  # [B,K,1,1]
        local_sum = torch.sum(w * neigh_local, dim=1)  # [B, L, D]
        return local_sum

    def fuse(self, orig, agg, neibor, weights):
        # Global vectors: [B, D]
        if orig.dim() == 2:
            concat = torch.cat([orig, agg], dim=1)  # [B,2D]
            enhanced = agg  # [B,D]
            gate = self.gate_global(concat)  # [B,D], sigmoid
            return gate * enhanced + (1 - gate) * orig

        # Local vectors: [B, L, D]
        elif orig.dim() == 3:
            B, K, L, D = neibor.shape
            concat = torch.cat([orig, agg], dim=2)  # [B, L, 2D]
            enhanced = agg
            gate = torch.sigmoid(self.gate_local(concat.view(B * L, -1))).view(B, L, D)
            return gate * enhanced + (1 - gate) * orig


    def forward(self,
                orig_global,    # [B, D]
                neigh_global,   # [B, K, D]
                orig_local,     # [B, L, D]
                neigh_local     # [B, K, L, D]
                ):
        # ANW
        weights = self.compute_weights(orig_global, neigh_global)  # [B,K]
        # fuse neighbor
        global_sum = self.aggregate_global(orig_global, neigh_global, weights)  # [B,D]
        local_sum = self.aggregate_local(orig_local, neigh_local, weights)  # [B, L, D]

        # RGA
        final_global = self.fuse(orig_global, global_sum, neigh_global, weights)  # [B,D]
        final_local = self.fuse(orig_local, local_sum, neigh_local, weights)  # [B,L,D] 或 [B,L,2D]

        return final_global, final_local
