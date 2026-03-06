import torch
import torch.nn as nn
import torch.nn.functional as Fy
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds
from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


'''
原来的NetworkEncoderRoPE的作用：
2D 关键点 → MLP → token
cliffcam / cam_angvel / imgseq 三个条件的 embedding（直接相加）
→ 12 层 EncoderRoPEBlock（RoPE+局部窗口）
→ final_layer 输出 151 维 pred_x（对应 EnDecoder 的 x_norm）
→ 额外用同一条特征 x 做 pred_cam 和 static_conf_logits。


双分支版本 NetworkEncoderRoPE
为了方便接入训练，接口和返回值都保持不变（pred_x / pred_cam / static_conf_logits），只是在内部拆成：
    Pose 分支：只吃 2D 关键点；
    Visual 分支：吃 bbox（cliffcam）+ 相机角速度 + 图像特征；
    Fuse 分支：把两路融合后，再走一段 Transformer，最后输出 151 维 pred_x。

改动点：

增加了三个超参数：pose_layers / vis_layers / fuse_layers（有默认值，对现有 cfg 兼容）。

新增：
self.pose_blocks、self.vis_blocks、self.fuse_blocks三个Transformer；
self.fuse_proj 把 pose+visual concat后投影回latent_dim；
分支：pose3d_coarse、root_vel_coarse、gv_orient_coarse、beta_coarse、delta_pose_coarse。
forward 里不再把条件直接加到 x 上，而是：
    x_pose：只由 2D 关键点生成 → Pose 分支 transformer；
    x_vis：由 cliffcam / cam_angvel / imgseq embedding 相加得到 → Visual 分支 transformer；
    x_fuse：concat[x_pose, x_vis] → Linear → Fuse 分支 transformer → 输出 pred_x 等。
现有训练代码只用到 pred_x / pred_cam / static_conf_logits / pred_context 这几个key，新加的几个 coarse 输出都是额外字段，理论不会影响现有逻辑。
    
'''
class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        output_dim=151,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # NEW: 三段Transformer的层数（默认加起来 = 原来的 num_layers）
        pose_layers=4,
        vis_layers=4,
        fuse_layers=None,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
    ):
        super().__init__()
        # input
        self.output_dim = output_dim
        self.max_len = max_len
        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        # 三段的层数
        self.pose_layers = pose_layers
        self.vis_layers = vis_layers
        if fuse_layers is None:
            # 默认让三段层数之和不小于原来的num_layers
            remain = max(1, num_layers - pose_layers - vis_layers)
            self.fuse_layers = remain
        else:
            self.fuse_layers = fuse_layers
        # ** Pose 分支输入（2D 关键点） **
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )
        # 条件 embedding（bbox/cam/img）——供 Visual 分支使用
        self._build_condition_embedder()
        # ** 三段 Transformer **
        # Pose分支
        self.pose_blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.pose_layers)
            ]
        )
        # Visual分支
        self.vis_blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.vis_layers)
            ]
        )
        # Fuse分支
        self.fuse_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.fuse_blocks = nn.ModuleList(
            [
                # 沿用gvhmr的rope和局部attention
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.fuse_layers)
            ]
        )
        # ** 输出头 **
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = pred_cam_dim > 0
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)
        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)
        # ** Pose分支中间监督head **
        self.pose3d_head = Mlp(self.latent_dim, out_features=17 * 3)
        self.pose_root_vel_head = Mlp(self.latent_dim, out_features=3)
        self.pose_gv_orient_head = Mlp(self.latent_dim, out_features=6)
        # ** Visual 分支中间监督 head **
        self.beta_head = Mlp(self.latent_dim, out_features=10)
        self.delta_pose_head = Mlp(self.latent_dim, out_features=72)
        self.avgbeta = avgbeta

    def _build_condition_embedder(self):
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

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        """
        Args:
            length: (B,)
            obs: (B, L, 17, 3)  2D 关键点 (x, y, conf)
            f_cliffcam: (B, L, 3), CLIFF-Cam 参数 (bbox / scale)
            f_cam_angvel: (B, L, 6), 相机角速度
            f_imgseq: (B, L, C), 图像特征 (HMR2 encoder 输出)
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # ** Pose 分支：只用 2D 关键点 **
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0.0  
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x_pose = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, latent_dim)

        # ** Visual 分支：cliffcam + cam_angvel + imgseq **
        vis_feats = []
        if f_cliffcam is not None:
            vis_feats.append(self.cliffcam_embedder(f_cliffcam))
        if f_cam_angvel is not None and hasattr(self, "cam_angvel_embedder"):
            vis_feats.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            vis_feats.append(self.imgseq_embedder(f_imgseq))
        if len(vis_feats) == 0:
            x_vis = torch.zeros_like(x_pose)
        else:
            x_vis = torch.stack(vis_feats, dim=0).sum(dim=0)  # (B, L, latent_dim)
        # ** attention **
        assert B == length.size(0)
        pmask, attnmask = self._build_attn_masks(length, L, device=x_pose.device)
        # ** Pose branch Transformer **
        for block in self.pose_blocks:
            x_pose = block(x_pose, attn_mask=attnmask, tgt_key_padding_mask=pmask)
        # ** Visual branch Transformer **
        for block in self.vis_blocks:
            x_vis = block(x_vis, attn_mask=attnmask, tgt_key_padding_mask=pmask)
        # Pose 分支中间输出（可选监督）
        pose3d_coarse = self.pose3d_head(x_pose).view(B, L, 17, 3)
        root_vel_coarse = self.pose_root_vel_head(x_pose)      # (B, L, 3)
        gv_orient_coarse = self.pose_gv_orient_head(x_pose)    # (B, L, 6)

        # Visual 分支中间输出（可选监督）
        beta_coarse = self.beta_head(x_vis)                    # (B, L, 10)
        delta_pose_coarse = self.delta_pose_head(x_vis)        # (B, L, 72)

        # ** Fuse branch：concat + 投影 + Transformer **
        x = torch.cat([x_pose, x_vis], dim=-1)                 # (B, L, 2*latent_dim)
        x = self.fuse_proj(x)                                  # (B, L, latent_dim)
        for block in self.fuse_blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)
        
        # ** 输出 x_norm（151维） **
        sample = self.final_layer(x)  # (B, L, 151)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, 10)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)
        # ** 额外输出：cam + static_conf **
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)  
        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')
        output = {
            "pred_context": x,             # 融合后的上下文特征
            "pred_x": sample,              # 151 维 x_norm
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            # 新增一些输出，便于给中间的transformer单独监督
            "pose3d_coarse": pose3d_coarse,
            "root_vel_coarse": root_vel_coarse,
            "gv_orient_coarse": gv_orient_coarse,
            "beta_coarse": beta_coarse,
            "delta_pose_coarse": delta_pose_coarse,
            "pose_context": x_pose,
            "vis_context": x_vis,
        }
        return output

# Add to MainStore
group_name = "network/gvhmr"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
