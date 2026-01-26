
import math
import torch
import torch.nn.functional as F
from torch import nn 
from functools import partial
import numpy as np
from einops import rearrange
from einops_exts import  rearrange_many
from timm.models.vision_transformer import Attention
from timm.models.layers import to_2tuple
from ddpm.BiFlowNet import BiFlowNet

# helpers functions

class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=(16,16,16,), patch_size=2, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT block.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4*hidden_size*2, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., eta=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        if eta is not None: # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(hidden_features), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(out_features), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        x = self.gamma1 * self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gamma2 * self.fc2(x)
        x = self.drop2(x)
        return x

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('grid_size:', grid_size)

    grid_x = np.arange(grid_size[0], dtype=np.float32)
    grid_y = np.arange(grid_size[1], dtype=np.float32)
    grid_z = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False,**block_kwargs):
        super().__init__()
        self.skip_linear = nn.Linear(2*hidden_size, hidden_size) if skip else None
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4 * hidden_size*2, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c , skip= None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x,skip], dim = -1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



def exists(x):
    return x is not None

def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 256
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, z, h, w = x.shape
        x = rearrange(x,'b c z x y -> b (z x y) c').contiguous()
        qkv = self.to_qkv(x).chunk(3, dim=2)
        q, k, v = rearrange_many(
            qkv, 'b d (h c) -> b h d c ', h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h (z x y) c -> b (h c) z x y ',z = z, x = h ,y = w ).contiguous()
        out = self.to_out(out)
        return out


class ControlledBiFlowNet(BiFlowNet):

    def forward(
        self,
        x,
        time,
        control,
        y=None,
        res=None
    ):
        assert (y is not None) == (
            self.cond_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        b = x.shape[0]
        ori_shape = (x.shape[2]*8,x.shape[3]*8,x.shape[4]*8) 
        x_IntraPatch = x.clone()
        p = self.sub_volume_size[0]
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)
        p1 , p2 , p3= x_IntraPatch.size(2) , x_IntraPatch.size(3) , x_IntraPatch.size(4)
        x_IntraPatch = rearrange(x_IntraPatch , 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')
        x = self.init_conv(x)
        r = x.clone()


        t = self.time_mlp(time) if exists(self.time_mlp) else None
        c = t.shape[-1]
        t_DiT = t.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)

        if self.cond_classes:
            assert y.shape == (x.shape[0],)
            cond_emb = self.cond_emb(y)
            cond_emb_DiT = cond_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t = t + cond_emb
            t_DiT = t_DiT + cond_emb_DiT
        if self.res_condition:
            if len(res.shape) == 1:
                res = res.unsqueeze(0)
            res_condition_emb = self.res_mlp(res)
            t = torch.cat((t,res_condition_emb),dim=1)
            res_condition_emb_DiT = res_condition_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t_DiT = torch.cat((t_DiT,res_condition_emb_DiT),dim=1)

        x_IntraPatch = self.x_embedder(x_IntraPatch)
        x_IntraPatch = x_IntraPatch + self.pos_embed
        h_DiT , h_Unet,h,=[],[],[]
        for Block, MlpLayer in self.IntraPatchFlow_input:
            x_IntraPatch = Block(x_IntraPatch,t_DiT)
            h_DiT.append(x_IntraPatch)
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)

        for Block in self.IntraPatchFlow_mid:
            x_IntraPatch = Block(x_IntraPatch,t_DiT)

        for Block, MlpLayer in self.IntraPatchFlow_output:
            x_IntraPatch = Block(x_IntraPatch,t_DiT , h_DiT.pop())
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)
        

        for idx, (block1, spatial_attn1, block2, spatial_attn2,downsample) in enumerate(self.downs):
            if idx <self.feature_fusion :
                x = x + h_Unet.pop(0)
            x = block1(x, t)
            x = spatial_attn1(x)
            h.append(x)
            x = block2(x, t)
            x = spatial_attn2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)
        x += control.pop()
        for idx, (block1, spatial_attn1,block2, spatial_attn2,  upsample) in enumerate(self.ups):
            if len(self.ups)-idx <= 2:
                x = x + h_Unet.pop(0)
            x = torch.cat((x, h.pop() + control.pop()), dim=1)
            x = block1(x, t)
            x = spatial_attn1(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = spatial_attn2(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
    
class ControlNet(nn.Module):
    def __init__(
        self,
        dim,
        hint_channel,
        learn_sigma = False,
        cond_classes=None,
        dim_mults=(1, 1, 2, 4, 8),
        sub_volume_size = (8,8,8),
        patch_size = 1,
        channels=3,
        attn_heads=8,
        init_dim=None,
        init_kernel_size=3,
        use_sparse_linear_attn=[0,0,0,1,1],
        resnet_groups=8,
        DiT_num_heads = 8,
        mlp_ratio=4,
        vq_size=64,
        res_condition=True,
        num_mid_DiT=1
    ):
        self.cond_classes = cond_classes
        self.res_condition=res_condition

        super().__init__()
        self.channels = channels    
        self.vq_size = vq_size
        out_dim = 2*channels if learn_sigma else channels
        self.dim = dim #
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(channels+hint_channel, init_dim, (init_kernel_size, init_kernel_size,
                                   init_kernel_size), padding=(init_padding, init_padding, init_padding))


        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.feature_fusion = np.asarray([item[0]==item[1] for item in in_out ]).sum()
        self.num_mid_DiT= num_mid_DiT
        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        if self.cond_classes is not None:
            self.cond_emb = nn.Embedding(cond_classes, time_dim)
        if self.res_condition is not None:
            self.res_mlp =nn.Sequential(nn.Linear(3, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
            time_dim = 2* time_dim
        # layers
        ### DiT blocks 
        self.sub_volume_size = sub_volume_size
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed_Voxel(sub_volume_size, patch_size, channels+hint_channel, dim, bias=True)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)
        self.IntraPatchFlow_input = nn.ModuleList()
        for i in range(self.feature_fusion):
            temp = [DiTBlock(dim, 
                     DiT_num_heads, 
                     mlp_ratio=mlp_ratio,
                     )]
            temp.append(FinalLayer(dim,self.patch_size,dim))
            self.IntraPatchFlow_input.append(nn.ModuleList(temp))
        self.IntraPatchFlow_input = nn.ModuleList(self.IntraPatchFlow_input)

        
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            is_first = ind < self.feature_fusion - 1
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[ind] else nn.Identity(),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[ind] else nn.Identity(),
                Downsample(dim_out) if not is_last and not is_first else nn.Identity(),
                self.make_zero_conv(dim_out)
            ]))



        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)


        self.mid_spatial_attn = Residual(PreNorm(mid_dim, AttentionBlock(
                    mid_dim, heads=attn_heads)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        self.mid_zero_conv = self.make_zero_conv(mid_dim)


    def forward(
        self,
        x,
        time,
        hint,
        y=None,
        res=None
    ):
        assert (y is not None) == (
            self.cond_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        b = x.shape[0]
        ori_shape = (x.shape[2]*8,x.shape[3]*8,x.shape[4]*8) 
        x = torch.cat((x,hint),dim = 1)
        x_IntraPatch = x.clone()
        p = self.sub_volume_size[0]
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)
        p1 , p2 , p3= x_IntraPatch.size(2) , x_IntraPatch.size(3) , x_IntraPatch.size(4)
        x_IntraPatch = rearrange(x_IntraPatch , 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        c = t.shape[-1]
        t_DiT = t.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
        if self.cond_classes:
            assert y.shape == (x.shape[0],)
            cond_emb = self.cond_emb(y)
            cond_emb_DiT = cond_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t = t + cond_emb
            t_DiT = t_DiT + cond_emb_DiT
        if self.res_condition:
            if len(res.shape) == 1:
                res = res.unsqueeze(0)
            res_condition_emb = self.res_mlp(res)
            t = torch.cat((t,res_condition_emb),dim=1)
            res_condition_emb_DiT = res_condition_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t_DiT = torch.cat((t_DiT,res_condition_emb_DiT),dim=1)

        x_IntraPatch = self.x_embedder(x_IntraPatch)
        x_IntraPatch = x_IntraPatch + self.pos_embed
        h_Unet,outs=[],[]
        
        for Block, MlpLayer in self.IntraPatchFlow_input:
            x_IntraPatch = Block(x_IntraPatch,t_DiT)
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)


        for idx, (block1, spatial_attn1, block2, spatial_attn2,downsample,zero_conv) in enumerate(self.downs):
            if idx <self.feature_fusion :
                x = x + h_Unet.pop(0)
            x = block1(x, t)
            x = spatial_attn1(x)
            x = block2(x, t)
            x = spatial_attn2(x)
            outs.append(zero_conv(x))
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)
        outs.append(self.mid_zero_conv(x))
        return outs
    

    def make_zero_conv(self, ch):
        return self.zero_module(nn.Conv3d(ch, ch, 1, padding=0))

    def zero_module(self,module):
        for p in module.parameters():
            p.detach().zero_()
        return module
    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.dim
        p = self.patch_size
        x,y,z = np.asarray(self.sub_volume_size) // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        volume = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return volume


class ControlLDM(nn.Module):
    def __init__(
        self,
        dim,
        learn_sigma = False,
        cond_classes=None,
        dim_mults=(1, 1, 2, 4, 8),
        sub_volume_size = (8, 8, 8),
        patch_size = 1,
        channels=3,
        attn_heads=8,
        init_dim=None,
        init_kernel_size=3,
        use_sparse_linear_attn=[0,0,0,1,1],
        resnet_groups=24, #
        DiT_num_heads = 8, #
        mlp_ratio=4,
        vq_size=64,
        res_condition=True,
        num_mid_DiT=1,
        control_scales = 1
    ):
        super().__init__()
        self.noise_estimator = ControlledBiFlowNet(
            dim,
            learn_sigma,
            cond_classes,
            dim_mults,
            sub_volume_size ,
            patch_size ,
            channels,
            attn_heads,
            init_dim,
            init_kernel_size,
            use_sparse_linear_attn,
            resnet_groups,
            DiT_num_heads,
            mlp_ratio,
            vq_size=vq_size,
            res_condition=res_condition,
            num_mid_DiT=num_mid_DiT,
        )

        self.controlnet = ControlNet(
            dim,
            hint_channel=channels,
            cond_classes=cond_classes,
            dim_mults=dim_mults,
            sub_volume_size=sub_volume_size,
            channels=channels,
            patch_size=patch_size,
            attn_heads=attn_heads,
            init_dim=init_dim,
            init_kernel_size=init_kernel_size,
            use_sparse_linear_attn=use_sparse_linear_attn,
            resnet_groups=resnet_groups,
            DiT_num_heads=DiT_num_heads,
            mlp_ratio=mlp_ratio,
            vq_size=vq_size
        )
        self.control_scales = [control_scales] * 6

    def forward(self, x, time, hint , y=None, res=None):
        control = self.controlnet(
            x = x , 
            time = time,
            hint = hint,
            y=y,
            res = res
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.noise_estimator(
            x = x,
            time = time,
            control = control,
            y = y,
            res = res
        )
        return eps

    @torch.no_grad()
    def load_pretrained_ldm(self, ldm_ckpt):
        checkpoint = torch.load(ldm_ckpt, map_location=torch.device('cpu'))
        self.noise_estimator.load_state_dict(checkpoint['ema'], strict=True)
        del checkpoint 
        self.noise_estimator.eval()
        for p in self.noise_estimator.parameters():
            p.requires_grad = False
        print('Checkpoint is load successfully!')
    
    @torch.no_grad()
    def load_controlnet_from_ckpt(self, ctnet_ckpt):
        checkpoint = torch.load(ctnet_ckpt, map_location=torch.device('cpu'))
        self.controlnet.load_state_dict(checkpoint['model'])
        
    def load_controlnet_from_noise_estimator(self):
        noise_estimator_sd = self.noise_estimator.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in noise_estimator_sd:
                this, target = scratch_sd[key], noise_estimator_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, d, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, d,h, w), dtype=target.dtype,device=target.device)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch



