import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------
# 1. Varied Defect Synthesis (VDS)
# --------------------------------------
class VariedDefectSynthesis:
    def __init__(self, params):
        self.params = params

    def saliency_scores(self, points, normals, patches):
        scores = []
        for p in patches:
            grads = [np.linalg.norm(normals[idx]) for idx in p]
            scores.append(np.mean(grads))
        return np.array(scores)

    def apply_bulge_dent(self, patch, normals):
        alpha = np.random.uniform(self.params['alpha_min'], self.params['alpha_max'])
        sigma = np.random.uniform(self.params['sigma_min'], self.params['sigma_max'])
        center = patch.mean(axis=0, keepdims=True)
        disp = patch - center
        r2 = np.sum(disp**2, axis=1, keepdims=True)
        weights = np.exp(-r2 / (sigma**2))
        return patch + alpha * normals * weights

    def apply_hole(self, patch, normals):
        radius = np.random.uniform(self.params['hole_radius_min'], self.params['hole_radius_max'])
        center = patch.mean(axis=0)
        distances = np.linalg.norm(patch - center, axis=1)
        mask = distances > radius
        new_patch = patch[mask]
        return new_patch

    def apply_crack(self, patch, normals):
        tau = np.random.uniform(*self.params['tau_range'])
        # remove points within slab
        d = np.dot(patch - patch.mean(axis=0), normals.mean(axis=0))
        mask = np.abs(d) > tau
        return patch[mask]

    def apply_roughness(self, patch, normals):
        eps = self.params['rough_eps']
        tangents = np.random.randn(*patch.shape)
        tangents -= (tangents * normals).sum(axis=1, keepdims=True) * normals
        tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        noise = eps * normals * np.random.randn(patch.shape[0],1) + eps * tangents * np.random.randn(patch.shape[0],1)
        return patch + noise

    def __call__(self, point_cloud, normals):
        B, N, _ = point_cloud.shape
        corrupted = []
        offsets = []
        for b in range(B):
            pts = point_cloud[b].cpu().numpy()
            nms = normals[b].cpu().numpy()
            # simple random patch selection
            patch_idx = np.random.choice(N, self.params['patch_size'], replace=False)
            patch = pts[patch_idx]
            n_patch = nms[patch_idx]
            op = np.random.choice(['bulge','hole','crack','rough'])
            if op=='bulge': new_patch = self.apply_bulge_dent(patch, n_patch)
            elif op=='hole': new_patch = self.apply_hole(patch, n_patch)
            elif op=='crack': new_patch = self.apply_crack(patch, n_patch)
            else: new_patch = self.apply_roughness(patch, n_patch)
            disp = new_patch - patch
            pts[patch_idx] = new_patch
            corrupted.append(pts)
            off = np.zeros_like(pts)
            off[patch_idx] = disp
            offsets.append(off)
        corrupted = torch.from_numpy(np.stack(corrupted)).float().to(point_cloud.device)
        offsets = torch.from_numpy(np.stack(offsets)).float().to(point_cloud.device)
        # global noise & drop
        corrupted += torch.randn_like(corrupted) * self.params['global_noise']
        return corrupted, offsets

# --------------------------------------
# 2. Backbone (Point-MAE)
# --------------------------------------
class PointMAEBackbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # placeholder: simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 384)
        )
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        ckpt = torch.load(path)
        self.mlp.load_state_dict(ckpt)

    def forward(self, patches, positions):
        # patches: [B, J, K, 3]
        B, J, K, _ = patches.shape
        x = patches.reshape(B*J*K, 3)
        feat = self.mlp(x).view(B, J, 384)
        return feat

# --------------------------------------
# 3. Voting Network
# --------------------------------------
class VotingHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.offset_head = nn.Linear(hidden_dim, 3)
        self.feature_head = nn.Linear(hidden_dim, in_dim)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.boundary_head = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = self.mlp(g)
        delta_x = self.offset_head(h)
        delta_f = self.feature_head(h)
        rho = F.softplus(self.scale_head(h))
        beta = torch.tanh(self.boundary_head(h))
        return delta_x, delta_f, rho.squeeze(-1), beta.squeeze(-1)

# --------------------------------------
# 4. Differentiable Clustering
# --------------------------------------
class DifferentiableClustering(nn.Module):
    def __init__(self, feat_dim, num_clusters=128):
        super().__init__()
        self.proto_mlp = nn.Linear(feat_dim + 4, num_clusters)

    def forward(self, votes, features):
        # votes: [B, J, 4], features: [B, J, D]
        x = torch.cat([votes, features], dim=-1)
        logits = self.proto_mlp(x)
        alpha = F.softmax(logits, dim=1)
        mu = (alpha.unsqueeze(-1) * votes).sum(dim=1) / (alpha.sum(dim=1, keepdim=True).unsqueeze(-1)+1e-6)
        h_k = (alpha.unsqueeze(-1) * features).sum(dim=1) / (alpha.sum(dim=1, keepdim=True).unsqueeze(-1)+1e-6)
        return mu, h_k, alpha

# --------------------------------------
# 5. Anomaly Localization Heads
# --------------------------------------
class PointScoringHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(feat_dim+4+feat_dim, 128), nn.ReLU(), nn.Linear(128,1)
        )

    def forward(self, p, mu_k, rho_k, h_k, g_i):
        # p: [N,3], mu_k:[K,3], rho_k:[N], h_k:[K,D], g_i:[N,D]
        # simple nearest-cluster scoring
        dists = torch.cdist(p, mu_k)  # [N,K]
        min_d, idx = dists.min(dim=1)
        h_k_sel = h_k[idx]
        rho_sel = rho_k[idx].unsqueeze(-1)
        inp = torch.cat([g_i, rho_sel, h_k_sel], dim=-1)
        a = self.score_mlp(inp).squeeze(-1)
        return a

# --------------------------------------
# 6. Full Model
# --------------------------------------
class Vote3DAnomalyDetector(nn.Module):
    def __init__(self, backbone, voting_head, clustering, scoring_head):
        super().__init__()
        self.backbone = backbone
        self.voting_head = voting_head
        self.clustering = clustering
        self.scoring_head = scoring_head

    def forward(self, point_cloud):
        B, N, _ = point_cloud.shape
        # trivial patching
        patches = point_cloud.unsqueeze(2)
        positions = point_cloud
        g = self.backbone(patches, positions)
        delta_x, delta_f, rho, beta = self.voting_head(g.view(-1, g.size(-1)))
        delta_x = delta_x.view(B,N,3)
        delta_f = delta_f.view(B,N,-1)
        rho = rho.view(B,N)
        votes = torch.cat([point_cloud+delta_x, rho.unsqueeze(-1)], dim=-1)
        mu, h_k, alpha = self.clustering(votes, delta_f)
        scores = self.scoring_head(point_cloud.view(-1,3), mu.view(-1,3), rho.view(-1), h_k.view(-1, h_k.size(-1)), g.view(-1, g.size(-1)))
        scores = scores.view(B,N)
        obj_scores = torch.max(torch.sigmoid(scores), dim=1)[0]
        # store offsets
        self.last_predicted_offsets = delta_x
        return scores, obj_scores
