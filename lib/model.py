# ============ lib/model.py ============
from .layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGeo(nn.Module):
    
    def __init__(self, dim_in, dim_z, dim_med, dim_out=2,
                 collaborative_mlp=True, proj_dim=30, 
                 pretrain_dim=None):
        super(AdaGeo, self).__init__()
        
        self.dim_in = dim_in
        self.dim_z = dim_z
        self.proj_dim = proj_dim
        self.collaborative_mlp = collaborative_mlp
        
        
        if pretrain_dim is None:
            pretrain_dim = dim_in
        
        self.pretrain_dim = pretrain_dim
        
        # ===== Determine transfer direction =====
        if dim_in > pretrain_dim:
            self.transfer_mode = "expand"
            self.shared_dim = pretrain_dim
            self.extra_dim = dim_in - pretrain_dim
            print(f"TrustGeo [Expand Mode]: {pretrain_dim}dim -> {dim_in}dim (extra {self.extra_dim}dim)")
            
        elif dim_in < pretrain_dim:
            self.transfer_mode = "shrink"
            self.shared_dim = dim_in
            self.extra_dim = pretrain_dim - dim_in
            print(f"TrustGeo [Shrink Mode]: {pretrain_dim}dim -> {dim_in}dim (missing {self.extra_dim}dim)")
            
        else:
          
            self.transfer_mode = "standard"
            self.shared_dim = dim_in
            self.extra_dim = 0
            print(f"TrustGeo [Standard Mode]: {dim_in}dim")
        
       
        self.att_attribute = SimpleAttention1(
            temperature=dim_z ** 0.5,
            d_q_in=pretrain_dim,
            d_k_in=pretrain_dim,
            d_v_in=pretrain_dim + 2,
            d_q_out=dim_z,
            d_k_out=dim_z,
            d_v_out=dim_z
        )
        self.w_1 = nn.Linear(pretrain_dim + 2, pretrain_dim + 2)
        self.w_2 = nn.Linear(pretrain_dim + 2, pretrain_dim + 2)
        
        #===== Transfer adaptation layers =====
        if self.transfer_mode == "expand":
            self.extra_encoder = nn.Sequential(
                nn.Linear(self.extra_dim, dim_z),
                nn.ReLU(),
                nn.Linear(dim_z, dim_z)
            )
            self.fusion = nn.Linear(pretrain_dim + 2 + dim_z, pretrain_dim + 2)
            
        elif self.transfer_mode == "shrink":
            self.input_proj = nn.Sequential(
                nn.Linear(dim_in, pretrain_dim),
                nn.ReLU(),
                nn.Linear(pretrain_dim, pretrain_dim)
            )
        
        # ===== Prediction Head =====
        self.final_dim = pretrain_dim + (pretrain_dim + 2) * 2
        
        if collaborative_mlp:
            self.pred = SimpleAttention2(
                temperature=dim_z ** 0.5,
                d_q_in=self.final_dim,
                d_k_in=pretrain_dim,
                d_v_in=2,
                d_q_out=dim_z,
                d_k_out=dim_z,
                d_v_out=2,
                drop_last_layer=False
            )
        else:
            self.pred = nn.Sequential(
                nn.Linear(self.final_dim, dim_med),
                nn.ReLU(),
                nn.Linear(dim_med, dim_out)
            )
        
    
        self.gamma_1 = nn.Parameter(torch.ones(1, 1))
        self.gamma_2 = nn.Parameter(torch.ones(1, 1))
        self.gamma_3 = nn.Parameter(torch.ones(1, 1))
        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1))

        
        

        self.residual_proj0 = nn.Linear(self.final_dim, proj_dim)
        self.residual_proj1 = nn.Linear(proj_dim, proj_dim)
        self.residual_proj2 = nn.Linear(proj_dim, proj_dim)

    def _adapt_features(self, lm_X, tg_X):
        """
        Adapt input features based on transfer mode
        Returns: Adapted (lm_X_adapted, tg_X_adapted, tg_X_extra)
        """
        if self.transfer_mode == "standard":
            return lm_X, tg_X, None
            
        elif self.transfer_mode == "expand":
            lm_X_shared = lm_X[:, :self.shared_dim]
            tg_X_shared = tg_X[:, :self.shared_dim]
            tg_X_extra = tg_X[:, self.shared_dim:]
            return lm_X_shared, tg_X_shared, tg_X_extra
            
        elif self.transfer_mode == "shrink":
            lm_X_proj = self.input_proj(lm_X)
            tg_X_proj = self.input_proj(tg_X)
            return lm_X_proj, tg_X_proj, None

    def forward(self, data):
       
        
        lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay = data
        
        N1 = lm_Y.size(0)
        N2 = tg_Y.size(0)
        device = lm_X.device
        
        # ===== Feature Adaptation =====
        lm_X_adapted, tg_X_adapted, tg_X_extra = self._adapt_features(lm_X, tg_X)
        
        
        if self.transfer_mode == "expand" and tg_X_extra is not None:
            tg_extra_feat = self.extra_encoder(tg_X_extra)
        else:
            tg_extra_feat = None
        
        # ===== Star-GNN =====
        ones = torch.ones(N1 + N2 + 1, device=device)
        lm_feature = torch.cat((lm_X_adapted, lm_Y), dim=1)
        tg_feature_0 = torch.cat((tg_X_adapted, torch.zeros(N2, 2, device=device)), dim=1)
        router_0 = torch.mean(lm_feature, dim=0, keepdim=True)
        all_feature_0 = torch.cat((lm_feature, tg_feature_0, router_0), dim=0)
        
        # ----- GNN Step 1 -----
        adj_matrix_0 = torch.diag(ones)
        
        delay_score = torch.exp(-self.gamma_1 * (self.alpha * lm_delay + self.beta))
        rou2tar_score_0 = torch.exp(-self.gamma_2 * (self.alpha * tg_delay + self.beta)).reshape(N2)
        
        _, attribute_score = self.att_attribute(tg_X_adapted, lm_X_adapted, lm_feature)
        attribute_score = torch.exp(attribute_score)
        
        adj_matrix_0[N1:N1 + N2, :N1] = attribute_score
        adj_matrix_0[-1, :N1] = delay_score
        adj_matrix_0[N1:N1 + N2, -1] = rou2tar_score_0
        
        degree_0 = torch.sum(adj_matrix_0, dim=1)
        degree_reverse_0 = 1.0 / (degree_0 + 1e-12)
        degree_matrix_reverse_0 = torch.diag(degree_reverse_0)
        
        degree_mul_adj_0 = degree_matrix_reverse_0 @ adj_matrix_0
        step_1_all_feature = self.w_1(degree_mul_adj_0 @ all_feature_0)
        
        tg_feature_1 = step_1_all_feature[N1:N1 + N2, :]
        router_1 = step_1_all_feature[-1, :].reshape(1, -1)
        
        # Fuse extra features (expand mode only)
        if self.transfer_mode == "expand" and tg_extra_feat is not None:
            tg_feature_1 = self.fusion(torch.cat([tg_feature_1, tg_extra_feat], dim=-1))
        
        # ----- GNN Step 2 -----
        adj_matrix_1 = torch.diag(ones)
        rou2tar_score_1 = torch.exp(-self.gamma_3 * (self.alpha * tg_delay + self.beta)).reshape(N2)
        adj_matrix_1[N1:N1 + N2, -1] = rou2tar_score_1
        
        all_feature_1 = torch.cat((lm_feature, tg_feature_1, router_1), dim=0)
        
        degree_1 = torch.sum(adj_matrix_1, dim=1)
        degree_reverse_1 = 1.0 / (degree_1 + 1e-12)
        degree_matrix_reverse_1 = torch.diag(degree_reverse_1)
        
        degree_mul_adj_1 = degree_matrix_reverse_1 @ adj_matrix_1
        step_2_all_feature = self.w_2(degree_mul_adj_1 @ all_feature_1)
        tg_feature_2 = step_2_all_feature[N1:N1 + N2, :]
        
        # ===== Final Feature Concatenation =====
        final_tg_feature = torch.cat((tg_X_adapted, tg_feature_1, tg_feature_2), dim=-1)
        
        # ===== Prediction =====
        if self.collaborative_mlp:
            y_pred, _ = self.pred(final_tg_feature, lm_X_adapted, lm_Y)
        else:
            y_pred = self.pred(final_tg_feature)
        
        return y_pred, final_tg_feature

    def get_feature_dim(self):
        """Return final feature dimension"""
        return self.final_dim
    
    def reset_parameters(self):
        """Reset all parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.gamma_1.data.fill_(1.0)
        self.gamma_2.data.fill_(1.0)
        self.gamma_3.data.fill_(1.0)
        self.alpha.data.fill_(1.0)
        self.beta.data.zero_()