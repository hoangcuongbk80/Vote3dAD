# network/Vote3D.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from vote3d_anomaly import (
    VariedDefectSynthesis,
    PointMAEBackbone,
    VotingHead,
    DifferentiableClustering,
    PointScoringHead,
    Vote3DAnomalyDetector
)


def model_fn(batch, model, cfg):
    """
    batch: dict with keys ['points', 'normals', 'labels']
    points: [B, N, 3]
    normals: [B, N, 3]
    labels: [B, N] binary ground truth
    """
    points = batch['points'].cuda(non_blocking=True)
    normals = batch['normals'].cuda(non_blocking=True)
    labels = batch['labels'].float().cuda(non_blocking=True)

    # 1) Synthesize varied defects
    vds = VariedDefectSynthesis(cfg.vds_params)
    corrupted_pc, offsets = vds(points, normals)

    # 2) Forward pass
    # model returns per-point anomaly scores [B, N]
    scores, obj_scores = model(corrupted_pc)

    # 3) Classification loss (point-level)
    bce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')

    # 4) Offset regression loss (smooth L1)
    # offsets: [B, N, 3]; model maybe returns predicted offsets in VotingHead
    # Here, assume model outputs predicted_offsets attr
    pred_offsets = model.last_predicted_offsets
    reg_loss = F.smooth_l1_loss(pred_offsets, offsets, reduction='mean')

    # Total loss
    loss = bce_loss + cfg.lambda_reg * reg_loss

    # Visualizations: pick a few examples
    visual_dict = {
        'input_pc': points[0].detach().cpu(),
        'corrupted_pc': corrupted_pc[0].detach().cpu(),
        'pred_scores': torch.sigmoid(scores[0]).detach().cpu()
    }

    # Meters
    meter_dict = {
        'loss': (loss.item(), points.size(0)),
        'bce_loss': (bce_loss.item(), points.size(0)),
        'reg_loss': (reg_loss.item(), points.size(0))
    }

    return loss, scores, visual_dict, meter_dict


class VoteAD(nn.Module):
    def __init__(self, in_channels, out_channels, cfg=None):
        super().__init__()
        # in_channels/out_channels unused for now
        # Instantiate components
        self.backbone = PointMAEBackbone(pretrained_path=cfg.pretrained_mae)
        self.voting_head = VotingHead(in_dim=cfg.mae_feat_dim)
        self.clustering = DifferentiableClustering(feat_dim=cfg.mae_feat_dim,
                                                   num_clusters=cfg.num_clusters)
        self.scoring_head = PointScoringHead(feat_dim=cfg.mae_feat_dim)

        # VDS for training
        self.vds = VariedDefectSynthesis(cfg.vds_params)

        # Store last predicted offsets for loss
        self.last_predicted_offsets = None

    def forward(self, point_cloud):
        """
        point_cloud: [B, N, 3]
        returns:
            scores: [B, N] raw logits of anomaly score
            obj_scores: [B] object-level anomaly confidence
        """
        B, N, _ = point_cloud.size()

        # 1) Compute normals externally or assume given
        # Here assume normals not needed for forward

        # 2) Partition into patches (not shown) and extract embeddings
        # For simplicity, treat each point as patch center
        # TODO: replace with actual patching
        patches = point_cloud.unsqueeze(2)  # [B, N, 1, 3]
        positions = point_cloud  # [B, N, 3]

        g = self.backbone(patches, positions)  # [B, N, D]

        # 3) Voting
        delta_x, delta_f, rho, beta = self.voting_head(g.view(-1, g.size(-1)))
        delta_x = delta_x.view(B, N, 3)
        delta_f = delta_f.view(B, N, -1)
        rho = rho.view(B, N)
        beta = beta.view(B, N)

        # Store offsets for loss
        self.last_predicted_offsets = delta_x

        votes = torch.cat([point_cloud + delta_x, rho.unsqueeze(-1)], dim=-1)  # [B, N, 4]

        # 4) Clustering
        mu, h_k, alpha = self.clustering(votes, delta_f)

        # 5) Score each point
        scores = []
        for b in range(B):
            p = point_cloud[b]  # [N, 3]
            g_i = g[b]           # [N, D]
            a_i = self.scoring_head(p, mu[b], rho[b], h_k[b], g_i)  # [N]
            scores.append(a_i)
        scores = torch.stack(scores, dim=0)  # [B, N]

        # 6) Object-level score: max pooling
        obj_scores = torch.max(torch.sigmoid(scores), dim=1)[0]

        return scores, obj_scores
