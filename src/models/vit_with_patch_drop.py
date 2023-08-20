import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from einops import rearrange

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
import random

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1,is_train=True):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        if is_train and random.random()>=0.7:
            patch_drop = torch.rand(x.shape[1])
            patch_drop[0] = True #keep cls token always in the first
            patch_drop = patch_drop > 0.5
            x = x[:,patch_drop,:]
        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i)
        x = self.norm(x)
        
        return x



def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)           #二维插值
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint


# class Model_test(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_layer = nn.Linear(1024*256, 256)

#     def forward(self,feature):
#         # feature_1 = feature.permute(0,2,1)
#         patches_fea_re = rearrange(feature,'b n1 d -> b 1 d n1')
#         # feature_2 = torch.zeros((32,1,8,256))
#         sim_matrix = torch.matmul(patches_fea_re,feature)
#         feature = rearrange(feature,'b n1 d -> b (n1 d)')
#         out = self.linear_layer(feature)
#         return sim_matrix


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Model_test().to(device)
#     summary(model, (1024,256))


    # model =VisionTransformer(
    #         img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
    #         mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    # summary(model, (3, 224, 224))
    """
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 768, 14, 14]         590,592
          Identity-2             [-1, 196, 768]               0
        PatchEmbed-3             [-1, 196, 768]               0
           Dropout-4             [-1, 197, 768]               0
         LayerNorm-5             [-1, 197, 768]           1,536
            Linear-6            [-1, 197, 2304]       1,771,776
           Dropout-7         [-1, 12, 197, 197]               0
            Linear-8             [-1, 197, 768]         590,592
           Dropout-9             [-1, 197, 768]               0
        Attention-10             [-1, 197, 768]               0
         Identity-11             [-1, 197, 768]               0
        LayerNorm-12             [-1, 197, 768]           1,536
           Linear-13            [-1, 197, 3072]       2,362,368
             GELU-14            [-1, 197, 3072]               0
          Dropout-15            [-1, 197, 3072]               0
           Linear-16             [-1, 197, 768]       2,360,064
          Dropout-17             [-1, 197, 768]               0
              Mlp-18             [-1, 197, 768]               0
         Identity-19             [-1, 197, 768]               0
            Block-20             [-1, 197, 768]               0
        LayerNorm-21             [-1, 197, 768]           1,536
           Linear-22            [-1, 197, 2304]       1,771,776
          Dropout-23         [-1, 12, 197, 197]               0
           Linear-24             [-1, 197, 768]         590,592
          Dropout-25             [-1, 197, 768]               0
        Attention-26             [-1, 197, 768]               0
         Identity-27             [-1, 197, 768]               0
        LayerNorm-28             [-1, 197, 768]           1,536
           Linear-29            [-1, 197, 3072]       2,362,368
             GELU-30            [-1, 197, 3072]               0
          Dropout-31            [-1, 197, 3072]               0
           Linear-32             [-1, 197, 768]       2,360,064
          Dropout-33             [-1, 197, 768]               0
              Mlp-34             [-1, 197, 768]               0
         Identity-35             [-1, 197, 768]               0
            Block-36             [-1, 197, 768]               0
        LayerNorm-37             [-1, 197, 768]           1,536
           Linear-38            [-1, 197, 2304]       1,771,776
          Dropout-39         [-1, 12, 197, 197]               0
           Linear-40             [-1, 197, 768]         590,592
          Dropout-41             [-1, 197, 768]               0
        Attention-42             [-1, 197, 768]               0
         Identity-43             [-1, 197, 768]               0
        LayerNorm-44             [-1, 197, 768]           1,536
           Linear-45            [-1, 197, 3072]       2,362,368
             GELU-46            [-1, 197, 3072]               0
          Dropout-47            [-1, 197, 3072]               0
           Linear-48             [-1, 197, 768]       2,360,064
          Dropout-49             [-1, 197, 768]               0
              Mlp-50             [-1, 197, 768]               0
         Identity-51             [-1, 197, 768]               0
            Block-52             [-1, 197, 768]               0
        LayerNorm-53             [-1, 197, 768]           1,536
           Linear-54            [-1, 197, 2304]       1,771,776
          Dropout-55         [-1, 12, 197, 197]               0
           Linear-56             [-1, 197, 768]         590,592
          Dropout-57             [-1, 197, 768]               0
        Attention-58             [-1, 197, 768]               0
         Identity-59             [-1, 197, 768]               0
        LayerNorm-60             [-1, 197, 768]           1,536
           Linear-61            [-1, 197, 3072]       2,362,368
             GELU-62            [-1, 197, 3072]               0
          Dropout-63            [-1, 197, 3072]               0
           Linear-64             [-1, 197, 768]       2,360,064
          Dropout-65             [-1, 197, 768]               0
              Mlp-66             [-1, 197, 768]               0
         Identity-67             [-1, 197, 768]               0
            Block-68             [-1, 197, 768]               0
        LayerNorm-69             [-1, 197, 768]           1,536
           Linear-70            [-1, 197, 2304]       1,771,776
          Dropout-71         [-1, 12, 197, 197]               0
           Linear-72             [-1, 197, 768]         590,592
          Dropout-73             [-1, 197, 768]               0
        Attention-74             [-1, 197, 768]               0
         Identity-75             [-1, 197, 768]               0
        LayerNorm-76             [-1, 197, 768]           1,536
           Linear-77            [-1, 197, 3072]       2,362,368
             GELU-78            [-1, 197, 3072]               0
          Dropout-79            [-1, 197, 3072]               0
           Linear-80             [-1, 197, 768]       2,360,064
          Dropout-81             [-1, 197, 768]               0
              Mlp-82             [-1, 197, 768]               0
         Identity-83             [-1, 197, 768]               0
            Block-84             [-1, 197, 768]               0
        LayerNorm-85             [-1, 197, 768]           1,536
           Linear-86            [-1, 197, 2304]       1,771,776
          Dropout-87         [-1, 12, 197, 197]               0
           Linear-88             [-1, 197, 768]         590,592
          Dropout-89             [-1, 197, 768]               0
        Attention-90             [-1, 197, 768]               0
         Identity-91             [-1, 197, 768]               0
        LayerNorm-92             [-1, 197, 768]           1,536
           Linear-93            [-1, 197, 3072]       2,362,368
             GELU-94            [-1, 197, 3072]               0
          Dropout-95            [-1, 197, 3072]               0
           Linear-96             [-1, 197, 768]       2,360,064
          Dropout-97             [-1, 197, 768]               0
              Mlp-98             [-1, 197, 768]               0
         Identity-99             [-1, 197, 768]               0
           Block-100             [-1, 197, 768]               0
       LayerNorm-101             [-1, 197, 768]           1,536
          Linear-102            [-1, 197, 2304]       1,771,776
         Dropout-103         [-1, 12, 197, 197]               0
          Linear-104             [-1, 197, 768]         590,592
         Dropout-105             [-1, 197, 768]               0
       Attention-106             [-1, 197, 768]               0
        Identity-107             [-1, 197, 768]               0
       LayerNorm-108             [-1, 197, 768]           1,536
          Linear-109            [-1, 197, 3072]       2,362,368
            GELU-110            [-1, 197, 3072]               0
         Dropout-111            [-1, 197, 3072]               0
          Linear-112             [-1, 197, 768]       2,360,064
         Dropout-113             [-1, 197, 768]               0
             Mlp-114             [-1, 197, 768]               0
        Identity-115             [-1, 197, 768]               0
           Block-116             [-1, 197, 768]               0
       LayerNorm-117             [-1, 197, 768]           1,536
          Linear-118            [-1, 197, 2304]       1,771,776
         Dropout-119         [-1, 12, 197, 197]               0
          Linear-120             [-1, 197, 768]         590,592
         Dropout-121             [-1, 197, 768]               0
       Attention-122             [-1, 197, 768]               0
        Identity-123             [-1, 197, 768]               0
       LayerNorm-124             [-1, 197, 768]           1,536
          Linear-125            [-1, 197, 3072]       2,362,368
            GELU-126            [-1, 197, 3072]               0
         Dropout-127            [-1, 197, 3072]               0
          Linear-128             [-1, 197, 768]       2,360,064
         Dropout-129             [-1, 197, 768]               0
             Mlp-130             [-1, 197, 768]               0
        Identity-131             [-1, 197, 768]               0
           Block-132             [-1, 197, 768]               0
       LayerNorm-133             [-1, 197, 768]           1,536
          Linear-134            [-1, 197, 2304]       1,771,776
         Dropout-135         [-1, 12, 197, 197]               0
          Linear-136             [-1, 197, 768]         590,592
         Dropout-137             [-1, 197, 768]               0
       Attention-138             [-1, 197, 768]               0
        Identity-139             [-1, 197, 768]               0
       LayerNorm-140             [-1, 197, 768]           1,536
          Linear-141            [-1, 197, 3072]       2,362,368
            GELU-142            [-1, 197, 3072]               0
         Dropout-143            [-1, 197, 3072]               0
          Linear-144             [-1, 197, 768]       2,360,064
         Dropout-145             [-1, 197, 768]               0
             Mlp-146             [-1, 197, 768]               0
        Identity-147             [-1, 197, 768]               0
           Block-148             [-1, 197, 768]               0
       LayerNorm-149             [-1, 197, 768]           1,536
          Linear-150            [-1, 197, 2304]       1,771,776
         Dropout-151         [-1, 12, 197, 197]               0
          Linear-152             [-1, 197, 768]         590,592
         Dropout-153             [-1, 197, 768]               0
       Attention-154             [-1, 197, 768]               0
        Identity-155             [-1, 197, 768]               0
       LayerNorm-156             [-1, 197, 768]           1,536
          Linear-157            [-1, 197, 3072]       2,362,368
            GELU-158            [-1, 197, 3072]               0
         Dropout-159            [-1, 197, 3072]               0
          Linear-160             [-1, 197, 768]       2,360,064
         Dropout-161             [-1, 197, 768]               0
             Mlp-162             [-1, 197, 768]               0
        Identity-163             [-1, 197, 768]               0
           Block-164             [-1, 197, 768]               0
       LayerNorm-165             [-1, 197, 768]           1,536
          Linear-166            [-1, 197, 2304]       1,771,776
         Dropout-167         [-1, 12, 197, 197]               0
          Linear-168             [-1, 197, 768]         590,592
         Dropout-169             [-1, 197, 768]               0
       Attention-170             [-1, 197, 768]               0
        Identity-171             [-1, 197, 768]               0
       LayerNorm-172             [-1, 197, 768]           1,536
          Linear-173            [-1, 197, 3072]       2,362,368
            GELU-174            [-1, 197, 3072]               0
         Dropout-175            [-1, 197, 3072]               0
          Linear-176             [-1, 197, 768]       2,360,064
         Dropout-177             [-1, 197, 768]               0
             Mlp-178             [-1, 197, 768]               0
        Identity-179             [-1, 197, 768]               0
           Block-180             [-1, 197, 768]               0
       LayerNorm-181             [-1, 197, 768]           1,536
          Linear-182            [-1, 197, 2304]       1,771,776
         Dropout-183         [-1, 12, 197, 197]               0
          Linear-184             [-1, 197, 768]         590,592
         Dropout-185             [-1, 197, 768]               0
       Attention-186             [-1, 197, 768]               0
        Identity-187             [-1, 197, 768]               0
       LayerNorm-188             [-1, 197, 768]           1,536
          Linear-189            [-1, 197, 3072]       2,362,368
            GELU-190            [-1, 197, 3072]               0
         Dropout-191            [-1, 197, 3072]               0
          Linear-192             [-1, 197, 768]       2,360,064
         Dropout-193             [-1, 197, 768]               0
             Mlp-194             [-1, 197, 768]               0
        Identity-195             [-1, 197, 768]               0
           Block-196             [-1, 197, 768]               0
       LayerNorm-197             [-1, 197, 768]           1,536
================================================================
Total params: 85,646,592
Trainable params: 85,646,592
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 408.53
Params size (MB): 326.72
Estimated Total Size (MB): 735.82
----------------------------------------------------------------
    """