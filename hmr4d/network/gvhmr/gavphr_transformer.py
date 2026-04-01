"""
GA-VPHR: Geometry-Aware Dual-Branch Cross-Attention Network

完全兼容GVHMR训练框架的实现版本
基于 fuze_transformer.py 结构，添加：
1. 骨架邻接先验 (Skeleton Adjacency Prior)
2. 运动学链编码 (Kinematic Chain Encoding)  
3. 几何感知交叉注意力 (Geometry-Aware Cross-Attention)
4. 门控融合 (Gated Fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from hmr4d.configs import MainStore, builds
from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


# ============================================================================
# 骨架定义 (COCO 17-joint format)
# ============================================================================

# 骨架连接边 (parent, child)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (0, 5), (0, 6),  # nose to shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hip connection
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# 父节点映射
PARENT_JOINT = {
    0: -1, 1: 0, 2: 0, 3: 1, 4: 2,
    5: 0, 6: 0, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 5, 12: 6, 13: 11, 14: 12, 15: 13, 16: 14,
}


def build_adjacency_matrix(num_joints: int = 17) -> torch.Tensor:
    """构建骨架邻接矩阵，初始化为解剖学连接"""
    A = torch.zeros(num_joints, num_joints)
    for i in range(num_joints):
        A[i, i] = 1.0
    for (i, j) in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # 二阶邻居
    A_sq = torch.mm(A, A)
    for i in range(num_joints):
        for j in range(num_joints):
            if A[i, j] == 0 and A_sq[i, j] > 0:
                A[i, j] = 0.5
    return A


def build_kinematic_relations(num_joints: int = 17) -> torch.Tensor:
    """构建运动学关系矩阵: 0=self, 1=parent, 2=child, 3=other"""
    R = torch.full((num_joints, num_joints), 3, dtype=torch.long)
    for i in range(num_joints):
        R[i, i] = 0
        if PARENT_JOINT[i] >= 0:
            R[i, PARENT_JOINT[i]] = 1
        for j in range(num_joints):
            if PARENT_JOINT.get(j, -1) == i:
                R[i, j] = 2
    return R


# ============================================================================
# 几何感知交叉注意力模块
# ============================================================================

class GeometryAwareCrossAttention(nn.Module):
    """
    几何感知交叉注意力 (GA-CA)
    
    公式 (论文 Sec 3.4.2):
        s_{ij} = <W_Q p_i, W_K v_j> / sqrt(d)
        b_{ij} = γ · Σ_k A_{ik} · sim(v_j, p_k)
        α_{ij} = Softmax_j(s_{ij} + b_{ij})
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1, geo_bias_scale=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.geo_bias_scale = geo_bias_scale
        
        # 标准交叉注意力投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = zero_module(nn.Linear(dim, dim))
        
        # 几何偏置的相似度投影
        self.sim_proj_q = nn.Linear(dim, dim // 4)
        self.sim_proj_k = nn.Linear(dim, dim // 4)
        
        # LayerNorm
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, time_adjacency=None, key_padding_mask=None):
        """
        Args:
            query: (B, L, D) - 查询分支特征
            key_value: (B, L, D) - 键值分支特征
            time_adjacency: (L, L) - 时序邻接矩阵（可选）
            key_padding_mask: (B, L) - padding mask
        """
        B, L, D = query.shape
        
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        
        # QKV投影
        q = self.q_proj(q_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 标准注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # === 几何感知偏置 ===
        if time_adjacency is not None:
            # 计算 sim(v_j, p_k)
            q_sim = self.sim_proj_q(q_norm)  # (B, L, D/4)
            k_sim = self.sim_proj_k(kv_norm)  # (B, L, D/4)
            sim_scale = (D // 4) ** -0.5
            sim_matrix = torch.matmul(k_sim, q_sim.transpose(-2, -1)) * sim_scale  # (B, L, L)
            
            # b_{ij} = γ · Σ_k A_{ik} · sim(v_j, p_k)
            # A: (L, L), sim_matrix: (B, L, L)
            geo_bias = torch.matmul(time_adjacency.unsqueeze(0), sim_matrix)  # (B, L, L)
            geo_bias = geo_bias * self.geo_bias_scale
            
            attn_scores = attn_scores + geo_bias.unsqueeze(1)
        
        # Mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        return out


class BidirectionalGACA(nn.Module):
    """
    双向几何感知交叉注意力 + 门控融合
    
    Pose ← Visual: 获取视觉上下文
    Visual ← Pose: 获取姿态信息
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1, geo_bias_scale=0.5):
        super().__init__()
        
        # 双向交叉注意力
        self.pose_cross_vis = GeometryAwareCrossAttention(dim, num_heads, dropout, geo_bias_scale)
        self.vis_cross_pose = GeometryAwareCrossAttention(dim, num_heads, dropout, geo_bias_scale)
        
        # 门控网络
        self.pose_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.vis_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # FFN
        self.pose_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.vis_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x_pose, x_vis, time_adjacency=None, key_padding_mask=None):
        """
        Args:
            x_pose: (B, L, D) - Pose分支特征
            x_vis: (B, L, D) - Visual分支特征
        Returns:
            x_pose_out, x_vis_out: 更新后的特征
        """
        # Pose queries Visual
        h_pose = self.pose_cross_vis(x_pose, x_vis, time_adjacency, key_padding_mask)
        # Visual queries Pose
        h_vis = self.vis_cross_pose(x_vis, x_pose, time_adjacency, key_padding_mask)
        
        # 门控融合
        g_pose = self.pose_gate(torch.cat([x_pose, h_pose], dim=-1))
        g_vis = self.vis_gate(torch.cat([x_vis, h_vis], dim=-1))
        
        x_pose_out = x_pose + g_pose * h_pose
        x_vis_out = x_vis + g_vis * h_vis
        
        # FFN
        x_pose_out = x_pose_out + self.pose_ffn(x_pose_out)
        x_vis_out = x_vis_out + self.vis_ffn(x_vis_out)
        
        return x_pose_out, x_vis_out


# ============================================================================
# 骨架感知关节编码器
# ============================================================================

class SkeletonAwareJointEncoder(nn.Module):
    """
    带骨架拓扑先验的关节编码器
    
    在关节级别进行自注意力，注入邻接矩阵偏置
    """
    
    def __init__(self, latent_dim=512, num_joints=17, num_heads=8, dropout=0.1, adj_bias_scale=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.adj_bias_scale = adj_bias_scale
        
        # 关节坐标编码
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )
        # 关节身份嵌入
        self.joint_embed = nn.Parameter(torch.randn(1, 1, num_joints, 32) * 0.02)
        # 置信度编码
        self.conf_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32),
        )
        
        # 关节特征投影
        self.joint_proj = nn.Linear(64, latent_dim)
        
        # 关节级自注意力
        self.joint_attn_q = nn.Linear(latent_dim, latent_dim)
        self.joint_attn_k = nn.Linear(latent_dim, latent_dim)
        self.joint_attn_v = nn.Linear(latent_dim, latent_dim)
        self.joint_attn_out = nn.Linear(latent_dim, latent_dim)
        self.joint_norm = nn.LayerNorm(latent_dim)
        
        # 可学习的邻接矩阵
        adj_init = build_adjacency_matrix(num_joints)
        adj_logits = torch.where(
            adj_init > 0,
            torch.log(adj_init / (1 - adj_init + 1e-6)),
            torch.full_like(adj_init, -5.0)
        )
        self.adjacency_logits = nn.Parameter(adj_logits)
        
        # 邻接偏置MLP: A_{ij} -> bias
        self.adj_bias_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, num_heads),
        )
        
        # 运动学关系编码
        kin_relations = build_kinematic_relations(num_joints)
        self.register_buffer('kinematic_relations', kin_relations)
        self.kin_embed = nn.Embedding(4, self.head_dim)
        
        # 聚合到帧级
        self.agg_mlp = Mlp(
            num_joints * latent_dim,
            hidden_features=latent_dim * 2,
            out_features=latent_dim,
            drop=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, obs):
        """
        Args:
            obs: (B, L, J, 3) - 2D关键点 (x, y, conf)
        Returns:
            frame_feat: (B, L, D) - 帧级特征
            joint_feat: (B, L, J, D) - 关节级特征（用于可视化）
        """
        B, L, J, _ = obs.shape
        
        coords = obs[..., :2]  # (B, L, J, 2)
        conf = obs[..., 2:3]   # (B, L, J, 1)
        visible = conf > 0.5
        
        # 坐标编码
        coord_feat = self.coord_mlp(coords)  # (B, L, J, 32)
        coord_feat = coord_feat + self.joint_embed
        coord_feat = coord_feat * visible + self.joint_embed * ~visible
        
        # 置信度编码
        conf_feat = self.conf_mlp(conf)  # (B, L, J, 32)
        
        # 拼接并投影
        joint_feat = torch.cat([coord_feat, conf_feat], dim=-1)  # (B, L, J, 64)
        joint_feat = self.joint_proj(joint_feat)  # (B, L, J, D)
        
        # === 关节级自注意力 + 骨架先验 ===
        # 重塑为 (B*L, J, D)
        joint_feat_flat = joint_feat.view(B * L, J, -1)
        joint_feat_norm = self.joint_norm(joint_feat_flat)
        
        q = self.joint_attn_q(joint_feat_norm).view(B * L, J, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.joint_attn_k(joint_feat_norm).view(B * L, J, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.joint_attn_v(joint_feat_norm).view(B * L, J, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 标准注意力分数
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*L, H, J, J)
        
        # 骨架邻接偏置
        A = torch.sigmoid(self.adjacency_logits)  # (J, J)
        adj_bias = self.adj_bias_mlp(A.unsqueeze(-1))  # (J, J, H)
        adj_bias = adj_bias.permute(2, 0, 1) * self.adj_bias_scale  # (H, J, J)
        attn_scores = attn_scores + adj_bias.unsqueeze(0)
        
        # 运动学链偏置
        kin_embed = self.kin_embed(self.kinematic_relations)  # (J, J, head_dim)
        kin_bias = torch.einsum('bhid,ijd->bhij', q, kin_embed)
        attn_scores = attn_scores + kin_bias
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        joint_out = torch.matmul(attn_weights, v)
        joint_out = joint_out.transpose(1, 2).reshape(B * L, J, -1)
        joint_out = self.joint_attn_out(joint_out)
        
        # 残差连接
        joint_feat_updated = joint_feat_flat + joint_out
        joint_feat_updated = joint_feat_updated.view(B, L, J, -1)
        
        # 聚合到帧级
        frame_feat = self.agg_mlp(joint_feat_updated.view(B, L, -1))
        
        return frame_feat, joint_feat_updated


# ============================================================================
# 主网络: GA-VPHR
# ============================================================================

class NetworkEncoderRoPE(nn.Module):
    """
    GA-VPHR: 几何感知双分支交叉注意力网络
    
    完全兼容GVHMR训练框架
    """
    
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
        # branch layers
        pose_layers=4,
        vis_layers=4,
        fuse_layers=None,
        # GA-CA config
        gaca_interval=2,
        geo_bias_scale=0.5,
        adj_bias_scale=1.0,
        use_skeleton_encoder=True,
        # output heads
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
    ):
        super().__init__()
        
        # 保存配置
        self.output_dim = output_dim
        self.max_len = max_len
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.avgbeta = avgbeta
        
        self.pose_layers = pose_layers
        self.vis_layers = vis_layers
        self.gaca_interval = gaca_interval
        self.use_skeleton_encoder = use_skeleton_encoder
        
        if fuse_layers is None:
            self.fuse_layers = max(1, num_layers - pose_layers - vis_layers)
        else:
            self.fuse_layers = fuse_layers
        
        # ==================== Pose分支输入 ====================
        if use_skeleton_encoder:
            # 使用骨架感知编码器
            self.skeleton_encoder = SkeletonAwareJointEncoder(
                latent_dim=latent_dim,
                num_joints=17,
                num_heads=num_heads,
                dropout=dropout,
                adj_bias_scale=adj_bias_scale,
            )
        else:
            # 使用原始编码（兼容）
            self.learned_pos_linear = nn.Linear(2, 32)
            self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
            self.embed_noisyobs = Mlp(
                17 * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout
            )
        
        # ==================== Visual分支输入 ====================
        self._build_condition_embedder()
        
        # ==================== Transformer块 ====================
        # Pose分支
        self.pose_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(pose_layers)
        ])
        
        # Visual分支
        self.vis_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(vis_layers)
        ])
        
        # GA-CA模块
        num_gaca = max(pose_layers, vis_layers) // gaca_interval
        self.gaca_blocks = nn.ModuleList([
            BidirectionalGACA(latent_dim, num_heads, dropout, geo_bias_scale)
            for _ in range(num_gaca)
        ])
        
        # Fuse分支
        self.fuse_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.fuse_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(self.fuse_layers)
        ])
        
        # ==================== 输出头 ====================
        self.final_layer = Mlp(latent_dim, out_features=output_dim)
        
        self.pred_cam_head = pred_cam_dim > 0
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)
        
        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)
        
        # ==================== 辅助输出头 ====================
        # Pose分支
        self.pose3d_head = Mlp(latent_dim, out_features=17 * 3)
        self.pose_root_vel_head = Mlp(latent_dim, out_features=3)
        self.pose_gv_orient_head = Mlp(latent_dim, out_features=6)
        
        # Visual分支
        self.beta_head = Mlp(latent_dim, out_features=10)
        self.delta_pose_head = Mlp(latent_dim, out_features=72)
        
    def _build_condition_embedder(self):
        """构建Visual分支的条件编码器"""
        latent_dim = self.latent_dim
        dropout = self.dropout
        
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
    
    def _build_attn_masks(self, length, L, device):
        """构建attention mask"""
        assert length.size(0) > 0
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
    
    def _build_time_adjacency(self, L, device, window=5):
        """构建时序邻接矩阵（用于GA-CA的几何偏置）"""
        adj = torch.zeros(L, L, device=device)
        for i in range(L):
            for j in range(max(0, i - window), min(L, i + window + 1)):
                adj[i, j] = 1.0 / (abs(i - j) + 1)
        return adj
    
    def _encode_pose_input(self, obs):
        """编码Pose分支输入"""
        B, L, J, C = obs.shape
        
        if self.use_skeleton_encoder:
            frame_feat, joint_feat = self.skeleton_encoder(obs)
            return frame_feat, joint_feat
        else:
            # 原始编码方式
            obs = obs.clone()
            visible_mask = obs[..., [2]] > 0.5
            obs[~visible_mask[..., 0]] = 0.0
            f_obs = self.learned_pos_linear(obs[..., :2])
            f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
            x_pose = self.embed_noisyobs(f_obs.view(B, L, -1))
            return x_pose, None
    
    def _encode_visual_input(self, f_cliffcam, f_cam_angvel, f_imgseq):
        """编码Visual分支输入"""
        vis_feats = []
        
        if f_cliffcam is not None:
            vis_feats.append(self.cliffcam_embedder(f_cliffcam))
        
        if f_cam_angvel is not None and hasattr(self, "cam_angvel_embedder"):
            vis_feats.append(self.cam_angvel_embedder(f_cam_angvel))
        
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            vis_feats.append(self.imgseq_embedder(f_imgseq))
        
        if len(vis_feats) == 0:
            raise ValueError("Visual分支至少需要一个输入")
        
        x_vis = torch.stack(vis_feats, dim=0).sum(dim=0)
        return x_vis
    
    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        """
        Args:
            length: (B,) 有效序列长度
            obs: (B, L, 17, 3) 2D关键点
            f_cliffcam: (B, L, 3) CLIFF相机参数
            f_cam_angvel: (B, L, 6) 相机角速度
            f_imgseq: (B, L, C) 图像特征
        
        Returns:
            dict: 与GVHMR完全兼容的输出字典
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3
        
        # ==================== 输入编码 ====================
        x_pose, joint_feat = self._encode_pose_input(obs)
        x_vis = self._encode_visual_input(f_cliffcam, f_cam_angvel, f_imgseq)
        
        # ==================== 构建Mask ====================
        pmask, attnmask = self._build_attn_masks(length, L, device=obs.device)
        time_adj = self._build_time_adjacency(L, obs.device)
        
        # ==================== 双分支 + 渐进式GA-CA ====================
        gaca_idx = 0
        max_layers = max(self.pose_layers, self.vis_layers)
        
        for i in range(max_layers):
            # Pose分支
            if i < self.pose_layers:
                x_pose = self.pose_blocks[i](x_pose, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            
            # Visual分支
            if i < self.vis_layers:
                x_vis = self.vis_blocks[i](x_vis, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            
            # GA-CA (每隔gaca_interval层)
            if (i + 1) % self.gaca_interval == 0 and gaca_idx < len(self.gaca_blocks):
                x_pose, x_vis = self.gaca_blocks[gaca_idx](
                    x_pose, x_vis, time_adj, pmask
                )
                gaca_idx += 1
        
        # ==================== 分支辅助输出 ====================
        pose3d_coarse = self.pose3d_head(x_pose).view(B, L, 17, 3)
        root_vel_coarse = self.pose_root_vel_head(x_pose)
        gv_orient_coarse = self.pose_gv_orient_head(x_pose)
        
        beta_coarse = self.beta_head(x_vis)
        delta_pose_coarse = self.delta_pose_head(x_vis)
        
        # ==================== Fusion ====================
        x = torch.cat([x_pose, x_vis], dim=-1)
        x = self.fuse_proj(x)
        
        for block in self.fuse_blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)
        
        # ==================== 主输出 ====================
        sample = self.final_layer(x)
        
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)
        
        # Camera预测
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)
        
        # Static confidence
        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)
        
        # ==================== 输出字典 ====================
        output = {
            # 主要输出（与GVHMR兼容）
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            
            # 分支特征
            "pose_context": x_pose,
            "vis_context": x_vis,
            
            # Pose分支辅助输出
            "pose3d_coarse": pose3d_coarse,
            "root_vel_coarse": root_vel_coarse,
            "gv_orient_coarse": gv_orient_coarse,
            
            # Visual分支辅助输出
            "beta_coarse": beta_coarse,
            "delta_pose_coarse": delta_pose_coarse,
        }
        
        # 如果使用骨架编码器，保存关节特征用于可视化
        if joint_feat is not None:
            output["joint_features"] = joint_feat
        
        return output


# ============================================================================
# 注册到MainStore
# ============================================================================

group_name = "network/gvhmr"
MainStore.store(
    name="gavphr_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)

# # 也可以用原名注册以直接替换
# MainStore.store(
#     name="relative_transformer",
#     node=builds(NetworkEncoderRoPE, populate_full_signature=True),
#     group=group_name,
# )
