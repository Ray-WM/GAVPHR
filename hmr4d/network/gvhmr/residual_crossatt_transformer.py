"""
Residual Cross-Attention Transformer (Plan A)

核心思想：保留完整的GVHMR baseline encoder作为主干，只学习一个cross-attention增量。
- baseline encoder (可加载预训练权重，前期冻结) → baseline_feat
- 轻量 visual branch (2层) → visual_feat  
- cross-attention(baseline_feat, visual_feat) → delta
- output = baseline_feat + gate * delta   (gate初始化为0)

这样保证：
1. camera-space指标不退化（baseline能力完整保留）
2. world-space轨迹质量可以通过cross-attention增量改善
3. 计算开销可控（只多了2层visual transformer + 1层cross-attention）
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


class GatedCrossAttention(nn.Module):
    """
    Cross-attention + zero-initialized gate.
    baseline_feat queries visual_feat, 输出一个残差delta.
    gate初始化为0，训练初期不干预baseline。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, hidden_features=dim * 4, out_features=dim, drop=dropout)
        
        # Zero-init gate: 训练初期delta≈0，模型行为≈baseline
        self.gate = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x_baseline, x_visual, key_padding_mask=None):
        """
        Args:
            x_baseline: (B, L, D) baseline encoder的输出特征
            x_visual:   (B, L, D) visual branch的输出特征
            key_padding_mask: (B, L) padding mask
        Returns:
            delta: (B, L, D) 残差增量 (已经乘过gate)
        """
        q = self.norm_q(x_baseline)
        kv = self.norm_kv(x_visual)
        
        attn_out, _ = self.cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
        )
        # Residual inside cross-attention block
        h = x_baseline + attn_out
        h = h + self.ffn(self.norm_ff(h))
        
        # Delta = cross-attention output - original input
        delta = h - x_baseline
        
        # Apply zero-init gate
        return self.gate * delta


class ResidualCrossAttEncoderRoPE(nn.Module):
    """
    方案A: Residual Baseline + Cross-Attention Delta
    
    结构:
        [所有输入] → 原始GVHMR encoder (12层, 可加载预训练权重)
                          ↓
                    baseline_feat (B, L, 512)
                          ↓
                    + gate * cross_attn_delta  ←── visual_branch (2层) 提供visual_feat
                          ↓                        cross_attention(baseline, visual) → delta
                    enhanced_feat (B, L, 512)
                          ↓
                    原始output heads → pred_x, pred_cam, static_conf
    
    加载权重策略:
        1. baseline部分 (learned_pos_linear, embed_noisyobs, condition_embedders, 
           blocks, final_layer, pred_cam_head, static_conf_head) 
           → 从baseline checkpoint加载
        2. visual_branch, cross_attention, gate → 随机初始化 (gate=0)
    """
    def __init__(
        self,
        # 与原始GVHMR完全一致的参数
        output_dim=151,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.1,
        avgbeta=True,
        # === 新增参数 ===
        cross_attn_insert_layers=None,  # 在哪些层之后插入cross-attention, 默认[5, 11]即第6和12层后
        freeze_baseline_epochs=0,       # 前N个epoch冻结baseline (可选, 由外部控制)
    ):
        super().__init__()
        
        # ==================== 存储参数 ====================
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
        
        # Cross-attention插入位置: 默认在第6层和第12层之后 (0-indexed: [5, 11])
        if cross_attn_insert_layers is None:
            cross_attn_insert_layers = [5, 11]
        self.cross_attn_insert_layers = set(cross_attn_insert_layers)
        self.num_ca = len(cross_attn_insert_layers)
        
        # ==================== Part 1: 原始GVHMR Baseline (完整保留) ====================
        # Input embedding (与原版完全一致)
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout
        )
        
        # Condition embedders (与原版完全一致)
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
        
        # Transformer blocks (与原版完全一致: 12层)
        self.blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads (与原版完全一致)
        self.final_layer = Mlp(latent_dim, out_features=output_dim)
        
        self.pred_cam_head = pred_cam_dim > 0
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)
        
        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)
        
        # ==================== Part 2: 新增 Visual Branch (轻量) ====================
        # 每个insertion point对应一个vis block + 一个cross-attention block
        # visual branch逐步细化，在每次cross-attention前先更新一次visual representation
        self.vis_branch_blocks = nn.ModuleList([
            EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(self.num_ca)
        ])
        
        # ==================== Part 3: Cross-Attention + Gate ====================
        self.cross_attn_blocks = nn.ModuleList([
            GatedCrossAttention(latent_dim, num_heads, dropout)
            for _ in range(self.num_ca)
        ])

    def _build_attn_mask(self, L, device):
        """构建local attention mask (与原版逻辑一致)"""
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
    
    def load_baseline_weights(self, ckpt_path, strict=False):
        """
        从baseline checkpoint加载GVHMR encoder的权重。
        
        用法:
            model = ResidualCrossAttEncoderRoPE(...)
            model.load_baseline_weights("path/to/baseline.ckpt")
        
        这会加载: learned_pos_linear, learned_pos_params, embed_noisyobs,
                  cliffcam_embedder, cam_angvel_embedder, imgseq_embedder,
                  blocks, final_layer, pred_cam_head, static_conf_head
        
        不会加载 (随机初始化): vis_branch_blocks, cross_attn_blocks
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # 提取encoder部分的state_dict
        # checkpoint格式通常是 {"state_dict": {"network.xxx": ...}} 
        # 需要根据你的实际checkpoint结构调整key前缀
        if "state_dict" in checkpoint:
            full_sd = checkpoint["state_dict"]
        else:
            full_sd = checkpoint
        
        # 找到encoder的key前缀 (可能是 "network." 或 "model.network." 等)
        # 尝试常见的前缀
        prefix = None
        for key in full_sd.keys():
            if "learned_pos_linear" in key:
                prefix = key.split("learned_pos_linear")[0]
                break
        
        if prefix is None:
            print("[WARNING] Could not find baseline encoder keys in checkpoint. Skipping load.")
            return
        
        # 构建映射: checkpoint key → 本模型的key
        baseline_sd = {}
        my_sd = self.state_dict()
        
        for ckpt_key, value in full_sd.items():
            if not ckpt_key.startswith(prefix):
                continue
            local_key = ckpt_key[len(prefix):]
            
            # 跳过visual branch和cross-attention (这些是新增的)
            if local_key.startswith("vis_branch_") or local_key.startswith("cross_attn_"):
                continue
            
            if local_key in my_sd:
                if my_sd[local_key].shape == value.shape:
                    baseline_sd[local_key] = value
                else:
                    print(f"[WARNING] Shape mismatch for {local_key}: "
                          f"ckpt={value.shape}, model={my_sd[local_key].shape}")
            # else: key在checkpoint中有但本模型中没有，跳过
        
        missing, unexpected = self.load_state_dict(baseline_sd, strict=False)
        
        # 报告
        n_loaded = len(baseline_sd)
        n_total = len(my_sd)
        vis_keys = [k for k in my_sd if k.startswith("vis_branch_") or k.startswith("cross_attn_")]
        print(f"[Baseline Load] Loaded {n_loaded}/{n_total} parameters from checkpoint.")
        print(f"[Baseline Load] {len(vis_keys)} new parameters (vis_branch + cross_attn) randomly initialized.")
        print(f"[Baseline Load] Gate initial values: all zeros (model starts as baseline).")
        
        return missing, unexpected
    
    def get_baseline_params(self):
        """返回baseline部分的参数 (用于差异化学习率)"""
        baseline_names = set()
        for name, _ in self.named_parameters():
            if not (name.startswith("vis_branch_") or name.startswith("cross_attn_")):
                baseline_names.add(name)
        return [p for n, p in self.named_parameters() if n in baseline_names]
    
    def get_new_params(self):
        """返回新增部分的参数 (visual branch + cross-attention)"""
        return [p for n, p in self.named_parameters() 
                if n.startswith("vis_branch_") or n.startswith("cross_attn_")]
    
    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        """
        接口与原始 NetworkEncoderRoPE 完全一致。
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3
        
        # ==================== Shared: 计算condition features (只算一次) ====================
        obs_input = obs.clone()
        visible_mask = obs_input[..., [2]] > 0.5  # (B, L, J, 1)
        obs_input[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs_input[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        
        cond_feats = []
        cond_feats.append(self.cliffcam_embedder(f_cliffcam))
        if hasattr(self, "cam_angvel_embedder"):
            cond_feats.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            cond_feats.append(self.imgseq_embedder(f_imgseq))
        
        # Masks
        pmask = ~length_to_mask(length, L)  # (B, L)
        attnmask = self._build_attn_mask(L, obs.device)
        
        # ==================== Part 1: Baseline Encoder + 中间层Cross-Attention ====================
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, latent_dim)
        for f_delta in cond_feats:
            x = x + f_delta
        
        # Visual branch初始化 (detach防止梯度干扰baseline embedders)
        v = sum(f.detach() for f in cond_feats)
        
        ca_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)
            
            # 在指定层之后插入cross-attention
            if i in self.cross_attn_insert_layers and ca_idx < self.num_ca:
                # 先更新visual representation
                v = self.vis_branch_blocks[ca_idx](v, attn_mask=attnmask, tgt_key_padding_mask=pmask)
                # 再做cross-attention: baseline queries visual
                x = x + self.cross_attn_blocks[ca_idx](x, v, key_padding_mask=pmask)
                ca_idx += 1
        
        # ==================== Part 2: Output (与原版完全一致) ====================
        sample = self.final_layer(x)  # (B, L, 151)
        
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)
        
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)
        
        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)
        
        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


# ==================== Register to MainStore ====================
group_name = "network/gvhmr"
MainStore.store(
    name="residual_crossatt_transformer",
    node=builds(ResidualCrossAttEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
