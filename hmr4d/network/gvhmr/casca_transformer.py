"""
CASCA: Confidence-Aware Selective Cross-Attention for Dual-Branch HMR

完整双分支架构，融合三大关键改进：
1. Pose branch用baseline权重初始化 (learned_pos_linear + embed_noisyobs + condition embedders)
2. 所有cross-modal gate零初始化 → 训练初期两分支独立
3. Confidence-Aware Dynamic Gating → 根据observation quality动态调节cross-attention强度

结构:
    [2D keypoints] → Pose Branch (4层, baseline初始化)
                          ↕ CASCA (每2层一次, confidence-aware gating)
    [visual+camera] → Visual Branch (4层)
                          ↓
                    Concat + Fuse Transformer (4层) → pred_x
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


# ==================== CASCA Module ====================

class DifficultyEstimator(nn.Module):
    """
    估计每帧的"difficulty score": 置信度低 或 相机运动大 → difficulty高 → cross-attention强
    输入: 关键点置信度 + camera angular velocity
    输出: difficulty ∈ (0, 1), shape (B, L, 1)
    """
    def __init__(self, cam_angvel_dim=6):
        super().__init__()
        # 输入: 17个关节的置信度(17) + cam_angvel的norm(1) = 18
        self.net = nn.Sequential(
            nn.Linear(18, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            # 不加sigmoid，后面手动加，方便初始化控制
        )
        # 初始化bias使得初始difficulty≈0.5 (中性)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, joint_conf, cam_angvel):
        """
        Args:
            joint_conf: (B, L, 17) 关节置信度
            cam_angvel: (B, L, 6) 相机角速度
        Returns:
            difficulty: (B, L, 1) ∈ (0, 1)
        """
        cam_speed = cam_angvel[..., :3].norm(dim=-1, keepdim=True)  # (B, L, 1)
        features = torch.cat([joint_conf, cam_speed], dim=-1)  # (B, L, 18)
        return torch.sigmoid(self.net(features))  # (B, L, 1)


class CASCABlock(nn.Module):
    """
    Confidence-Aware Selective Cross-Attention Block
    
    双向cross-attention + confidence-aware dynamic gating + zero-init safety gate
    
    difficulty高时(遮挡/大运动): pose分支大量吸收visual信息
    difficulty低时(正常帧):      cross-attention几乎不生效, 保持独立性
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Pose ← Visual cross-attention
        self.norm_pose_q = nn.LayerNorm(dim)
        self.norm_vis_kv = nn.LayerNorm(dim)
        self.pose_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_pose_ff = nn.LayerNorm(dim)
        self.pose_ffn = Mlp(dim, hidden_features=dim * 4, out_features=dim, drop=dropout)
        
        # Visual ← Pose cross-attention
        self.norm_vis_q = nn.LayerNorm(dim)
        self.norm_pose_kv = nn.LayerNorm(dim)
        self.vis_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_vis_ff = nn.LayerNorm(dim)
        self.vis_ffn = Mlp(dim, hidden_features=dim * 4, out_features=dim, drop=dropout)
        
        # Zero-init safety gates (训练初期=0, 双分支完全独立)
        self.pose_gate = nn.Parameter(torch.zeros(1, 1, dim))
        self.vis_gate = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x_pose, x_vis, difficulty, key_padding_mask=None):
        """
        Args:
            x_pose: (B, L, D)
            x_vis:  (B, L, D) 
            difficulty: (B, L, 1) ∈ (0, 1) from DifficultyEstimator
            key_padding_mask: (B, L)
        Returns:
            x_pose_out, x_vis_out
        """
        # === Pose ← Visual ===
        q = self.norm_pose_q(x_pose)
        kv = self.norm_vis_kv(x_vis)
        attn_out, _ = self.pose_cross_attn(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask)
        delta_pose = attn_out + self.pose_ffn(self.norm_pose_ff(x_pose + attn_out)) - x_pose
        
        # === Visual ← Pose ===
        q2 = self.norm_vis_q(x_vis)
        kv2 = self.norm_pose_kv(x_pose)
        attn_out2, _ = self.vis_cross_attn(query=q2, key=kv2, value=kv2, key_padding_mask=key_padding_mask)
        delta_vis = attn_out2 + self.vis_ffn(self.norm_vis_ff(x_vis + attn_out2)) - x_vis
        
        # === Confidence-Aware Gating ===
        # difficulty高 → pose大量吸收visual (遮挡时需要visual帮忙)
        # difficulty低 → pose少量吸收visual (正常帧保持独立)
        pose_weight = difficulty          # (B, L, 1)
        vis_weight = 1.0 - difficulty     # 反向: pose可靠时visual多吸收pose信息
        
        # ========== E3①: 记录gate值和update norm ==========
        if not self.training:
            self._last_pose_gate_val = (self.pose_gate * difficulty).detach()  # (B, L, 1) 或 scalar
            self._last_vis_gate_val = (self.vis_gate * (1 - difficulty)).detach()
            self._last_pose_update_norm = (pose_weight * self.pose_gate * delta_pose).detach().norm(dim=-1, keepdim=True)
            self._last_vis_update_norm = (vis_weight * self.vis_gate * delta_vis).detach().norm(dim=-1, keepdim=True)       
        # 乘上zero-init safety gate
        x_pose_out = x_pose + pose_weight * self.pose_gate * delta_pose
        x_vis_out = x_vis + vis_weight * self.vis_gate * delta_vis
        
        return x_pose_out, x_vis_out


# ==================== Main Architecture ====================

class CASCAEncoderRoPE(nn.Module):
    """
    CASCA Dual-Branch Encoder
    
    结构:
        Pose Branch: [2D keypoints → baseline encoding] → 4层Transformer
        Visual Branch: [cliffcam + cam_angvel + imgseq] → 4层Transformer  
        CASCA: 每2层做一次confidence-aware bidirectional cross-attention
        Fuse: concat → proj → 4层Transformer → output heads
    
    加载baseline权重:
        Pose branch的input encoding (learned_pos_linear, embed_noisyobs)
        + condition embedders → 从baseline checkpoint加载
    """
    def __init__(
        self,
        # 与原始GVHMR一致的参数
        output_dim=151,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=512,
        num_layers=12,  # 保留兼容性, 实际用下面的分支层数
        num_heads=8,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.1,
        avgbeta=True,
        # === 双分支参数 ===
        pose_layers=4,
        vis_layers=4,
        fuse_layers=4,
        cross_attn_interval=2,  # 每隔几层做一次CASCA
        #ablation_const_gate=False,    # 新增
        #ablation_no_conf=False,       # 新增  
        #ablation_no_zero_init=False,  # 新增
    ):
        super().__init__()

        #self.ablation_const_gate = ablation_const_gate
        #self.ablation_no_conf = ablation_no_conf
        #self.ablation_no_zero_init = ablation_no_zero_init        

        self.output_dim = output_dim
        self.max_len = max_len
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.avgbeta = avgbeta
        self.pose_layers = pose_layers
        self.vis_layers = vis_layers
        self.fuse_layers = fuse_layers
        self.cross_attn_interval = cross_attn_interval
        
        # ==================== Pose Branch Input (与baseline一致, 可加载权重) ====================
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout
        )
        
        # ==================== Visual Branch Input ====================
        # 用与baseline相同结构的condition embedders (可加载权重)
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(imgseq_dim),
                zero_module(nn.Linear(imgseq_dim, latent_dim)),
            )
        
        # ==================== Pose Branch Transformer ====================
        self.pose_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(pose_layers)
        ])
        
        # ==================== Visual Branch Transformer ====================
        self.vis_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(vis_layers)
        ])
        
        # ==================== CASCA Modules ====================
        num_casca = max(pose_layers, vis_layers) // cross_attn_interval
        self.difficulty_estimator = DifficultyEstimator(cam_angvel_dim)
        self.casca_blocks = nn.ModuleList([
            CASCABlock(latent_dim, num_heads, dropout)
            for _ in range(num_casca)
        ])
        
        # 变体C: 非零初始化gate
        import os
        if os.environ.get("ABLATION_NO_ZERO_INIT", "0") == "1":
            for block in self.casca_blocks:
                nn.init.constant_(block.pose_gate, 0.5)
                nn.init.constant_(block.vis_gate, 0.5)        

        # ==================== Fuse Branch ====================
        self.fuse_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(fuse_layers)
        ])
        
        # ==================== Output Heads ====================
        self.final_layer = Mlp(latent_dim, out_features=output_dim)
        
        self.pred_cam_head = pred_cam_dim > 0
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)
        
        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)
    
    def _build_attn_mask(self, L, device):
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
        return attnmask
    
    def load_baseline_weights(self, ckpt_path):
        """
        从baseline checkpoint加载pose branch和condition embedder的权重。
        
        映射关系 (baseline → CASCA):
            learned_pos_linear → learned_pos_linear  (完全一致)
            learned_pos_params → learned_pos_params
            embed_noisyobs → embed_noisyobs
            cliffcam_embedder → cliffcam_embedder
            cam_angvel_embedder → cam_angvel_embedder
            imgseq_embedder → imgseq_embedder
            blocks.0-3 → pose_blocks.0-3  (baseline前4层 → pose branch)
            final_layer → final_layer
            pred_cam_head → pred_cam_head
            static_conf_head → static_conf_head
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            full_sd = checkpoint["state_dict"]
        else:
            full_sd = checkpoint
        
        # 找encoder key前缀
        prefix = None
        for key in full_sd.keys():
            if "learned_pos_linear" in key:
                prefix = key.split("learned_pos_linear")[0]
                break
        if prefix is None:
            print("[WARNING] Could not find baseline encoder keys. Skipping.")
            return
        
        my_sd = self.state_dict()
        loaded_sd = {}
        
        for ckpt_key, value in full_sd.items():
            if not ckpt_key.startswith(prefix):
                continue
            local_key = ckpt_key[len(prefix):]
            
            # 直接匹配的key (input embeddings, condition embedders, output heads)
            if local_key in my_sd and my_sd[local_key].shape == value.shape:
                loaded_sd[local_key] = value
                continue
            
            # baseline的 blocks.N → pose_blocks.N (前pose_layers层)
            if local_key.startswith("blocks."):
                layer_idx = int(local_key.split(".")[1])
                if layer_idx < self.pose_layers:
                    new_key = local_key.replace(f"blocks.{layer_idx}", f"pose_blocks.{layer_idx}")
                    if new_key in my_sd and my_sd[new_key].shape == value.shape:
                        loaded_sd[new_key] = value
        
        missing, unexpected = self.load_state_dict(loaded_sd, strict=False)
        
        n_loaded = len(loaded_sd)
        n_total = len(my_sd)
        new_keys = [k for k in my_sd if k not in loaded_sd]
        print(f"[CASCA Load] Loaded {n_loaded}/{n_total} params from baseline.")
        print(f"[CASCA Load] {len(new_keys)} new params randomly initialized.")
        print(f"[CASCA Load] CASCA gates: all zeros. Difficulty estimator: neutral init.")
        
    def get_pose_branch_params(self):
        """Pose branch参数 (用于差异化学习率)"""
        pose_prefixes = ("learned_pos_", "embed_noisyobs", "pose_blocks")
        return [p for n, p in self.named_parameters() 
                if any(n.startswith(pf) for pf in pose_prefixes)]
    
    def get_new_params(self):
        """新增模块参数: visual branch + CASCA + fuse + difficulty"""
        pose_prefixes = ("learned_pos_", "embed_noisyobs", "pose_blocks")
        output_prefixes = ("final_layer", "pred_cam_head", "static_conf_head", "pred_cam_mean", "pred_cam_std")
        cond_prefixes = ("cliffcam_embedder", "cam_angvel_embedder", "imgseq_embedder")
        old_prefixes = pose_prefixes + output_prefixes + cond_prefixes
        return [p for n, p in self.named_parameters()
                if not any(n.startswith(pf) for pf in old_prefixes)]
    
    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        B, L, J, C = obs.shape
        assert J == 17 and C == 3
        
        # ==================== Input Encoding ====================
        # Pose branch input (与baseline一致)
        obs_input = obs.clone()
        visible_mask = obs_input[..., [2]] > 0.5
        obs_input[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs_input[..., :2])
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x_pose = self.embed_noisyobs(f_obs.view(B, L, -1))
        
        # Visual branch input
        vis_feats = []
        vis_feats.append(self.cliffcam_embedder(f_cliffcam))
        if hasattr(self, "cam_angvel_embedder"):
            vis_feats.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            vis_feats.append(self.imgseq_embedder(f_imgseq))
        x_vis = sum(vis_feats)
        
        # ==================== Difficulty Score ====================
        joint_conf = obs[..., 2]  # (B, L, 17)
#       difficulty = self.difficulty_estimator(joint_conf, f_cam_angvel)  # (B, L, 1)
        import os
        if os.environ.get("ABLATION_CONST_GATE", "0") == "1":
            difficulty = torch.ones(B, L, 1, device=obs.device) * 0.5
        elif os.environ.get("ABLATION_NO_CONF", "0") == "1":
            dummy_conf = torch.zeros_like(joint_conf)
            dummy_angvel = torch.zeros_like(f_cam_angvel)
            difficulty = self.difficulty_estimator(dummy_conf, dummy_angvel)
        else:
            difficulty = self.difficulty_estimator(joint_conf, f_cam_angvel)       

        # ==================== Masks ====================
        pmask = ~length_to_mask(length, L)
        attnmask = self._build_attn_mask(L, obs.device)
        
        # ==================== Dual Branch + CASCA ====================
        casca_idx = 0
        max_layers = max(self.pose_layers, self.vis_layers)
        
        for i in range(max_layers):
            if i < self.pose_layers:
                x_pose = self.pose_blocks[i](x_pose, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            if i < self.vis_layers:
                x_vis = self.vis_blocks[i](x_vis, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            
            # CASCA every cross_attn_interval layers
            if (i + 1) % self.cross_attn_interval == 0 and casca_idx < len(self.casca_blocks):
                x_pose, x_vis = self.casca_blocks[casca_idx](
                    x_pose, x_vis, difficulty, key_padding_mask=pmask
                )
                casca_idx += 1
        
        # ==================== Fuse ====================
        x_fuse = torch.cat([x_pose, x_vis], dim=-1)
        x_fuse = self.fuse_proj(x_fuse)
        
        for block in self.fuse_blocks:
            x_fuse = block(x_fuse, attn_mask=attnmask, tgt_key_padding_mask=pmask)
        
        # ==================== Output ====================
        sample = self.final_layer(x_fuse)
        
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)
        
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x_fuse)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)
        
        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x_fuse)
        
        output = {
            "pred_context": x_fuse,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            # 额外输出 (用于分析, 不影响训练)
            "difficulty": difficulty,        # (B, L, 1) 可视化用
            "pose_context": x_pose,
            "vis_context": x_vis,
        }

        # ========== E3①: 收集所有CASCA block的gate信息 ==========
        if not self.training:
            gate_info = {}
            for i, block in enumerate(self.casca_blocks):
                if hasattr(block, '_last_pose_gate_val'):
                    gate_info[f'casca_{i}_pose_gate'] = block._last_pose_gate_val
                    gate_info[f'casca_{i}_vis_gate'] = block._last_vis_gate_val
                    gate_info[f'casca_{i}_pose_update_norm'] = block._last_pose_update_norm
                    gate_info[f'casca_{i}_vis_update_norm'] = block._last_vis_update_norm
            output['gate_info'] = gate_info

        return output


# ==================== Register ====================
group_name = "network/gvhmr"
MainStore.store(
    name="casca_transformer",
    node=builds(CASCAEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
