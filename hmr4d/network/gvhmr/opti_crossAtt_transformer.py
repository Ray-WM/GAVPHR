"""
改进版双分支Transformer

1. 交叉注意力：让Pose分支能够query Visual分支的信息，解决深度歧义
2. 渐进式融合：每隔几层进行一次信息交换，而不是最后才融合
3. 改进中间监督：Pose分支监督2D相关任务，避免病态的纯2D预测3D
4. 加入时序建模增强

结构：
[2D关键点] → Pose Transformer ←─────┐
                 ↓                   │ Cross Attention (每N层)
[视觉特征] → Visual Transformer ────┘
                 ↓
            Fuse Transformer → pred_x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds
from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, hidden_features=dim * 4, out_features=dim, drop=dropout)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x_query, x_key_value, key_padding_mask=None):
        # Cross Attention
        x_query_norm = self.norm1(x_query)
        x_kv_norm = self.norm2(x_key_value)
        
        attn_out, _ = self.cross_attn(
            query=x_query_norm,
            key=x_kv_norm,
            value=x_kv_norm,
            key_padding_mask=key_padding_mask
        )
        x = x_query + attn_out
        x = x + self.ffn(self.norm3(x))
        return x


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.pose_cross_attn = CrossAttentionBlock(dim, num_heads, dropout)
        self.vis_cross_attn = CrossAttentionBlock(dim, num_heads, dropout)
        self.pose_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.vis_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x_pose, x_vis, key_padding_mask=None):
        # Pose queries Visual (获取视觉上下文来解决深度歧义)
        x_pose_cross = self.pose_cross_attn(x_pose, x_vis, key_padding_mask)
        # Visual queries Pose (获取姿态信息来辅助轨迹估计)
        x_vis_cross = self.vis_cross_attn(x_vis, x_pose, key_padding_mask)
        # 门控融合
        pose_gate = self.pose_gate(torch.cat([x_pose, x_pose_cross], dim=-1))
        vis_gate = self.vis_gate(torch.cat([x_vis, x_vis_cross], dim=-1))
        
        x_pose_out = x_pose + pose_gate * (x_pose_cross - x_pose)
        x_vis_out = x_vis + vis_gate * (x_vis_cross - x_vis)
        return x_pose_out, x_vis_out


class TemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, hidden_features=dim * 2, out_features=dim, drop=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        # Conv1D expects (B, D, L)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        # Mask padding positions
        if mask is not None:
            x_conv = x_conv.masked_fill(mask.unsqueeze(-1), 0)
        x = residual + self.dropout(x_conv)
        x = x + self.ffn(self.norm(x))
        return x


class ImprovedNetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # output
        output_dim=151,
        max_len=120,
        # condition dims
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        # architecture
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # 分支配置
        pose_layers=4,
        vis_layers=4,
        fuse_layers=4,
        # 交叉注意力配置
        cross_attn_interval=2,  # 每隔几层做一次交叉注意力
        use_bidirectional_cross=True,  # 是否使用双向交叉注意力
        # 时序卷积
        use_temporal_conv=True,
        temporal_kernel_size=5,
        # output heads
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
    ):
        super().__init__()
        
        # Save config
        self.output_dim = output_dim
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.avgbeta = avgbeta
        
        self.pose_layers = pose_layers
        self.vis_layers = vis_layers
        self.fuse_layers = fuse_layers
        self.cross_attn_interval = cross_attn_interval
        self.use_bidirectional_cross = use_bidirectional_cross
        # ==================== Input Embeddings ====================
        # Pose分支：2D关键点编码
        self.joint_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        # 可学习的关节位置编码
        self.joint_pos_embed = nn.Parameter(torch.randn(1, 1, 17, 32) * 0.02)
        # 关节置信度编码
        self.conf_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32),
        )
        # 融合关节特征
        self.pose_input_proj = Mlp(
            17 * 64,  # 32 (pos) + 32 (conf) per joint
            hidden_features=latent_dim * 2,
            out_features=latent_dim,
            drop=dropout
        )
        # Visual分支：条件编码
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self._build_condition_embedder()
        
        # ==================== Transformer Blocks ====================
        # Pose分支 Transformer
        self.pose_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(pose_layers)
        ])
        # Visual分支 Transformer
        self.vis_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(vis_layers)
        ])
        # CrossAttention
        num_cross_attn = max(pose_layers, vis_layers) // cross_attn_interval
        if use_bidirectional_cross:
            self.cross_attn_blocks = nn.ModuleList([
                BidirectionalCrossAttention(latent_dim, num_heads, dropout)
                for _ in range(num_cross_attn)
            ])
        else:
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(latent_dim, num_heads, dropout)
                for _ in range(num_cross_attn)
            ])
        self.use_temporal_conv = use_temporal_conv
        if use_temporal_conv:
            self.pose_temporal_conv = TemporalConvBlock(latent_dim, temporal_kernel_size, dropout)
            self.vis_temporal_conv = TemporalConvBlock(latent_dim, temporal_kernel_size, dropout)
        self.fuse_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.fuse_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(fuse_layers)
        ])
        
        # ==================== Output Heads ====================
        self.final_layer = Mlp(latent_dim, out_features=output_dim)
        # Camera prediction
        self.pred_cam_head = None
        if pred_cam_dim > 0:
            self.pred_cam_head = Mlp(latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]))
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]))
        # Static confidence
        self.static_conf_head = None
        if static_conf_dim > 0:
            self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)

        # ==================== 中间监督头 ====================      
        # Pose分支：监督2D相关任务（不直接预测3D）
        self.pose2d_refine_head = Mlp(latent_dim, out_features=17 * 2)  # 2D关键点refinement
        self.joint_confidence_head = Mlp(latent_dim, out_features=17)   # 关节置信度
        self.pose_velocity_head = Mlp(latent_dim, out_features=17 * 2)  # 2D关键点速度
        # Visual分支：监督全局信息
        self.beta_head = Mlp(latent_dim, out_features=10)       # 体型参数
        self.cam_motion_head = Mlp(latent_dim, out_features=6)  # 相机运动
        # Fuse后的3D监督（用融合特征而非单一分支）
        self.pose3d_head = Mlp(latent_dim, out_features=17 * 3)     # 3D关键点
        self.root_vel_head = Mlp(latent_dim, out_features=3)        # 根节点速度
        self.orient_head = Mlp(latent_dim, out_features=6)          # 全局朝向
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        
        # CLIFF camera parameters
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # Camera angular velocity
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
            )
        
        # Image sequence features
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                nn.Linear(self.imgseq_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
    
    def _build_attn_masks(self, length, L, device):
        pmask = ~length_to_mask(length, L)
        
        if L > self.max_len:
            attnmask = torch.ones((L, L), device=device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None
            
        return pmask, attnmask
    
    def _encode_pose_input(self, obs):
        B, L, J, _ = obs.shape
        # 分离坐标和置信度
        coords = obs[..., :2]        # (B, L, 17, 2)
        conf = obs[..., 2:3]         # (B, L, 17, 1)
        # 可见性mask
        visible = conf > 0.5         # (B, L, 17, 1)
        # 编码坐标
        coords_feat = self.joint_encoder(coords)  # (B, L, 17, 32)
        # 加上可学习的关节位置编码
        coords_feat = coords_feat + self.joint_pos_embed
        # 不可见关节使用学习到的默认特征
        coords_feat = coords_feat * visible + self.joint_pos_embed * ~visible
        # 编码置信度
        conf_feat = self.conf_encoder(conf)  # (B, L, 17, 32)
        # 拼接
        joint_feat = torch.cat([coords_feat, conf_feat], dim=-1)  # (B, L, 17, 64)
        joint_feat = joint_feat.view(B, L, -1)  # (B, L, 17*64)
        # 投影到latent_dim
        x_pose = self.pose_input_proj(joint_feat)  # (B, L, latent_dim)
        return x_pose
    
    def _encode_visual_input(self, f_cliffcam, f_cam_angvel, f_imgseq):
        vis_feats = []
        if f_cliffcam is not None:
            vis_feats.append(self.cliffcam_embedder(f_cliffcam))            
        if f_cam_angvel is not None and hasattr(self, 'cam_angvel_embedder'):
            vis_feats.append(self.cam_angvel_embedder(f_cam_angvel))           
        if f_imgseq is not None and hasattr(self, 'imgseq_embedder'):
            vis_feats.append(self.imgseq_embedder(f_imgseq))        
        if len(vis_feats) == 0:
            raise ValueError("至少需要一个视觉输入")
        
        # 相加融合
        x_vis = torch.stack(vis_feats, dim=0).sum(dim=0)
        
        return x_vis
    
    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        B, L, J, C = obs.shape
        assert J == 17 and C == 3
        
        # ==================== 输入编码 ====================
        x_pose = self._encode_pose_input(obs)
        x_vis = self._encode_visual_input(f_cliffcam, f_cam_angvel, f_imgseq)
        
        # ==================== 构建Mask ====================
        pmask, attnmask = self._build_attn_masks(length, L, device=obs.device)
        
        # ==================== 并行双分支 + 渐进式交叉注意力 ====================
        cross_attn_idx = 0
        max_layers = max(self.pose_layers, self.vis_layers)
        
        for i in range(max_layers):
            # Pose分支
            if i < self.pose_layers:
                x_pose = self.pose_blocks[i](x_pose, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            
            # Visual分支
            if i < self.vis_layers:
                x_vis = self.vis_blocks[i](x_vis, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            
            # 交叉注意力（每隔cross_attn_interval层）
            if (i + 1) % self.cross_attn_interval == 0 and cross_attn_idx < len(self.cross_attn_blocks):
                if self.use_bidirectional_cross:
                    x_pose, x_vis = self.cross_attn_blocks[cross_attn_idx](x_pose, x_vis, pmask)
                else:
                    # 单向：Pose queries Visual
                    x_pose = self.cross_attn_blocks[cross_attn_idx](x_pose, x_vis, pmask)
                cross_attn_idx += 1
        
        # ==================== 时序卷积增强 ====================
        if self.use_temporal_conv:
            x_pose = self.pose_temporal_conv(x_pose, pmask)
            x_vis = self.vis_temporal_conv(x_vis, pmask)
        
        # ==================== 中间输出（分支级别） ====================
        # Pose分支：2D相关监督
        pose2d_refine = self.pose2d_refine_head(x_pose).view(B, L, 17, 2)
        joint_confidence = self.joint_confidence_head(x_pose)  # (B, L, 17)
        pose_velocity = self.pose_velocity_head(x_pose).view(B, L, 17, 2)
        
        # Visual分支：全局信息监督
        beta_pred = self.beta_head(x_vis)       # (B, L, 10)
        cam_motion = self.cam_motion_head(x_vis)  # (B, L, 6)
        
        # ==================== Fuse分支 ====================
        x_fuse = torch.cat([x_pose, x_vis], dim=-1)  # (B, L, 2*latent_dim)
        x_fuse = self.fuse_proj(x_fuse)              # (B, L, latent_dim)
        
        for block in self.fuse_blocks:
            x_fuse = block(x_fuse, attn_mask=attnmask, tgt_key_padding_mask=pmask)
        
        # ==================== Fuse后的3D监督 ====================
        pose3d_pred = self.pose3d_head(x_fuse).view(B, L, 17, 3)
        root_vel_pred = self.root_vel_head(x_fuse)   # (B, L, 3)
        orient_pred = self.orient_head(x_fuse)       # (B, L, 6)
        
        # ==================== 主输出 ====================
        pred_x = self.final_layer(x_fuse)  # (B, L, 151)
        
        # Beta平均化（可选）
        if self.avgbeta:
            betas = (pred_x[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            pred_x = torch.cat([pred_x[..., :126], betas, pred_x[..., 136:]], dim=-1)
        
        # Camera预测
        pred_cam = None
        if self.pred_cam_head is not None:
            pred_cam = self.pred_cam_head(x_fuse)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)
        
        # Static confidence
        static_conf_logits = None
        if self.static_conf_head is not None:
            static_conf_logits = self.static_conf_head(x_fuse)
        
        # ==================== 输出字典 ====================
        output = {
            # 主要输出（与原版兼容）
            "pred_x": pred_x,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            "pred_context": x_fuse,
            
            # 分支特征（用于分析）
            "pose_context": x_pose,
            "vis_context": x_vis,
            
            # Pose分支中间监督（2D相关）
            "pose2d_refine": pose2d_refine,       # (B, L, 17, 2) 2D关键点refinement
            "joint_confidence": joint_confidence,  # (B, L, 17) 关节置信度预测
            "pose_velocity": pose_velocity,        # (B, L, 17, 2) 2D速度
            
            # Visual分支中间监督
            "beta_coarse": beta_pred,    # (B, L, 10) 体型参数
            "cam_motion": cam_motion,    # (B, L, 6) 相机运动
            
            # Fuse后的3D监督
            "pose3d_fused": pose3d_pred,     # (B, L, 17, 3) 3D关键点
            "root_vel_fused": root_vel_pred, # (B, L, 3) 根节点速度
            "orient_fused": orient_pred,     # (B, L, 6) 全局朝向
            
            # 为了兼容你原来的代码，保留这些key
            "pose3d_coarse": pose3d_pred,
            "root_vel_coarse": root_vel_pred,
            "gv_orient_coarse": orient_pred,
            "delta_pose_coarse": torch.zeros(B, L, 72, device=obs.device),  # placeholder
        }
        
        return output


class AuxiliaryLosses(nn.Module):
    """
    辅助损失函数，用于中间监督
    """
    def __init__(self, 
                 pose2d_weight=1.0,
                 velocity_weight=0.5,
                 confidence_weight=0.1,
                 beta_weight=0.1,
                 pose3d_weight=1.0,
                 temporal_smooth_weight=0.5):
        super().__init__()
        self.pose2d_weight = pose2d_weight
        self.velocity_weight = velocity_weight
        self.confidence_weight = confidence_weight
        self.beta_weight = beta_weight
        self.pose3d_weight = pose3d_weight
        self.temporal_smooth_weight = temporal_smooth_weight
    
    def forward(self, outputs, targets, mask=None):
        losses = {}
        
        # 2D关键点refinement损失
        if "pose2d_gt" in targets and "pose2d_refine" in outputs:
            pose2d_loss = F.mse_loss(
                outputs["pose2d_refine"],
                targets["pose2d_gt"],
                reduction='none'
            )
            if mask is not None:
                pose2d_loss = (pose2d_loss * mask[..., None, None]).sum() / mask.sum()
            else:
                pose2d_loss = pose2d_loss.mean()
            losses["pose2d_loss"] = self.pose2d_weight * pose2d_loss
        
        # 2D速度损失（时序一致性）
        if "pose_velocity" in outputs:
            pred_vel = outputs["pose_velocity"]
            if "pose2d_gt" in targets:
                gt_vel = targets["pose2d_gt"][:, 1:] - targets["pose2d_gt"][:, :-1]
                pred_vel_trim = pred_vel[:, :-1]
                vel_loss = F.mse_loss(pred_vel_trim, gt_vel, reduction='none')
                if mask is not None:
                    mask_vel = mask[:, :-1]
                    vel_loss = (vel_loss * mask_vel[..., None, None]).sum() / mask_vel.sum()
                else:
                    vel_loss = vel_loss.mean()
                losses["velocity_loss"] = self.velocity_weight * vel_loss
        
        # 关节置信度损失
        if "joint_confidence" in outputs and "joint_visible" in targets:
            conf_loss = F.binary_cross_entropy_with_logits(
                outputs["joint_confidence"],
                targets["joint_visible"].float(),
                reduction='none'
            )
            if mask is not None:
                conf_loss = (conf_loss * mask[..., None]).sum() / mask.sum()
            else:
                conf_loss = conf_loss.mean()
            losses["confidence_loss"] = self.confidence_weight * conf_loss
        
        # Beta损失
        if "beta_gt" in targets and "beta_coarse" in outputs:
            beta_loss = F.mse_loss(outputs["beta_coarse"], targets["beta_gt"], reduction='none')
            if mask is not None:
                beta_loss = (beta_loss * mask[..., None]).sum() / mask.sum()
            else:
                beta_loss = beta_loss.mean()
            losses["beta_loss"] = self.beta_weight * beta_loss
        
        # 3D关键点损失（用融合后的特征）
        if "pose3d_gt" in targets and "pose3d_fused" in outputs:
            pose3d_loss = F.mse_loss(
                outputs["pose3d_fused"],
                targets["pose3d_gt"],
                reduction='none'
            )
            if mask is not None:
                pose3d_loss = (pose3d_loss * mask[..., None, None]).sum() / mask.sum()
            else:
                pose3d_loss = pose3d_loss.mean()
            losses["pose3d_loss"] = self.pose3d_weight * pose3d_loss
        
        # 时序平滑损失
        if self.temporal_smooth_weight > 0 and "pose3d_fused" in outputs:
            pred_3d = outputs["pose3d_fused"]
            # 加速度平滑
            vel = pred_3d[:, 1:] - pred_3d[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            smooth_loss = acc.pow(2).mean()
            losses["temporal_smooth_loss"] = self.temporal_smooth_weight * smooth_loss
        
        return losses


# Register to MainStore
group_name = "network/gvhmr"
MainStore.store(
    name="improved_relative_transformer",
    node=builds(ImprovedNetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
