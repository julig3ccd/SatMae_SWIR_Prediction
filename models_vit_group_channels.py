# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial


from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image



import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class GroupChannelsVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, channel_embed=256, num_channels=13,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)), **kwargs):
        super().__init__(**kwargs)
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']
        print("embed_dim as parameter for patch_embed", embed_dim)
        self.channel_groups = channel_groups

        self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                          for group in channel_groups])
        # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed[0].num_patches

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # Extra embedding for cls to fill embed_dim
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm      

        print("pos_embed in init", self.pos_embed.shape)
        print("channel_embed in init", self.channel_embed.shape)

        self.conv2d_decode = nn.Conv2d(1024, 2, kernel_size=1)
        # self.head = nn.Conv2d(embed_dim, out_channels=num_channels, kernel_size=1)
        # torch.nn.init.trunc_normal_(self.head.weight, std=0.02)
        # self.head.bias.data.fill_(0)    

    def forward_features(self, x):
        b, c, h, w = x.shape

        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

        #print("length x_c_embed", len(x_c_embed)) = 3 Groups

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        print("channel_embed", channel_embed.shape)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)
        print("pos_embed", pos_embed.shape)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
        print("channel_embed after expand", channel_embed.shape)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
        print("pos_embed after expand", pos_embed.shape)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)
        print("pos_channel", pos_channel.shape)
         

        # 96px (input size) /  8 (batch size) = 12 px (patch size)
        # 12px * 12px = 144px per patch (144 tokens)
        # 3 groups * 144 tokens = 432 tokens
        # 432 tokens + 1 global token = 433 tokens (+1 is propbably classification, is empty in the beginning and later predicted by model --> see https://www.researchgate.net/figure/Example-of-an-architecture-of-the-ViT-based-on-1_fig1_370462929)
        
        #TODO from each token/pixel with 1024 dim go back to 2 dim (for swir channels), or 13 dim (for whole picture)
        #1. find out what eally happens in pos_embed
        #2. research how decoding is done in UNET --> https://www.geeksforgeeks.org/u-net-architecture-explained/
        #3. find out where + 1 global token comes from and if it is set to end or beginning of sequence 

        #hint: probably the last (-1 global token) 144 tokens are the ones in our SWIR group 
        # try to reconstruct the channels from those 144 (last third of token sequence)
        # maybe try to stack always rows of 12 on top in a 2D layer until it is 12x12



        

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, G, L, D) # (batch, group, token seqeuence length, dim)
        print("before x.view(b,-1,D)",x.shape) #shape = ([8, 3, 144, 1024])
        x = x.view(b, -1, D)  # (N, G*L, D) # 3 groups (G) * 144 tokens (L) = 432 tokens
        print("after pos embed",x.shape) #shape ([8, 432, 1024]) 
        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)


        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D) concat the cls_token to the beginning of the sequence (seq is first dim)
        print("after concat cls_tokens, x" , x.shape)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        #shape ([8, 433, 1024])    
        print("after blocks",x.shape)    

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            print("after norm", x.shape)
            #outcome = x[:, 0]
            #print("outcome shape" , outcome.shape, outcome)
            
            #outcome_head = self.head(x.view(b, h, w, -1).permute(0, 3, 1, 2))
            #print("outcome with modified head shape", outcome_head.shape)
        print("input for reshape", x.shape)
        swir_tokens = x[:, -144:, :]
        # Reshape to [batch_size, 2, 12, 12, 8, 8]
        reshaped_tokens = swir_tokens.view(8, 12, 12, 1024)

        reshaped_tokens = reshaped_tokens.permute(0, 3, 1, 2)

        # print("before type conversion", reshaped_tokens.dtype)

        # reshaped_tokens = reshaped_tokens.to(torch.float)
        
        # print("after type conversion", reshaped_tokens.dtype)
        
        
        # Finally reshape to [batch_size, 2, 96, 96]
        final_image = self.conv2d_decode(reshaped_tokens)
        print("final_image shape", final_image.shape)


        #remove this printing block after test 
        image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())

    # Convert to PIL Image
    # If the tensor has multiple channels, convert each channel separately
        to_pil_image = transforms.ToPILImage()
    # Convert each channel to a PIL image and show
        for i in range(image_to_show.size(0)):  # Loop through channels
            channel_image = to_pil_image(image_to_show[i])
            channel_image.show(title=f'Channel {i}')

    # If you want to use matplotlib for showing multiple channels together:

    # Convert tensor to numpy array
        image_np = image_to_show.permute(1, 2, 0).numpy()  # Shape: [96, 96, 2]

    # Display the image using matplotlib
        plt.imshow(image_np)
        plt.title('Image with 2 Channels')
        plt.axis('off')  # Turn off axis
        plt.show()



        return final_image
    

# def reshape_to_image(outcome, h, w, patch_size):
#     batch_size, num_tokens, patch_dim = outcome.shape
    
#     # Exclude the cls token
#     #outcome = outcome[:, 1:, :]  # shape now (batch_size, num_patches, patch_dim)
    
#     # Determine the number of patches along height and width
#     num_patches = (h // patch_size) * (w // patch_size)
    
#     assert num_tokens - 1 == num_patches, "Mismatch between number of tokens and patches"
    
#     # Reshape to (batch_size, num_patches_height, num_patches_width, patch_dim)
#     outcome = outcome.view(batch_size, h // patch_size, w // patch_size, patch_dim)
    
#     # Permute to (batch_size, patch_dim, num_patches_height, num_patches_width)
#     outcome = outcome.permute(0, 3, 1, 2)
    
#     # Final reshape to (batch_size, num_channels, height, width)
#     num_channels = patch_dim
#     outcome = outcome.view(batch_size, num_channels, h, w)
    
#     return outcome  


def vit_base_patch16(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = GroupChannelsVisionTransformer(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = GroupChannelsVisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model