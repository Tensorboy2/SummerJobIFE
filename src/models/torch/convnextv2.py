import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN(nn.Module):
    '''
    Global Response Normalization layer.
    '''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
    
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=(2,3), keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class LayerNorm2d(nn.Module):
    '''
    Layer norm method to not have to permute manually all the time.
    '''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x

class ConvNeXtBlock(nn.Module):
    '''
    ConvNeXt v2 block using GRN.
    '''
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # Could use timm DropPath
    
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXtV2Encoder(nn.Module):
    '''
    ConvNeXt V2 Encoder for MAE.
    '''
    def __init__(self, in_chans=3, depths=[3,3,9,3], dims=[96,192,384,768]):
        super().__init__()
        self.depths = depths
        self.dims = dims
        
        # Downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)
        
        # Feature extraction stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
    
    def forward(self, x):
        features = []
        for i, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = downsample(x)
            x = stage(x)
            features.append(x)
        return x, features

class ConvNeXtV2Decoder(nn.Module):
    '''
    ConvNeXt V2 Decoder for MAE reconstruction.
    '''
    def __init__(self, encoder_dims=[96,192,384,768], decoder_dims=[512,256,128,64], out_chans=12):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        
        # Projection from encoder to decoder
        self.encoder_to_decoder = nn.Conv2d(encoder_dims[-1], decoder_dims[0], kernel_size=1)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[i], decoder_dims[i+1], 
                                 kernel_size=2, stride=2),
                LayerNorm2d(decoder_dims[i+1], eps=1e-6)
            )
            self.upsample_layers.append(upsample_layer)
        
        # Final upsampling to match input resolution
        self.final_upsample = nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], 
                                               kernel_size=4, stride=4)
        
        # Decoder stages (fewer blocks than encoder)
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=decoder_dims[i]) for _ in range(2)]  # Fewer blocks
            )
            self.decoder_stages.append(stage)
        
        # Final stage for the last decoder dimension
        final_stage = nn.Sequential(
            *[ConvNeXtBlock(dim=decoder_dims[-1]) for _ in range(2)]
        )
        self.decoder_stages.append(final_stage)
        
        # Output projection
        self.output_proj = nn.Conv2d(decoder_dims[-1], out_chans, kernel_size=1)
        
    def forward(self, x):
        # Project from encoder to decoder space
        x = self.encoder_to_decoder(x)
        
        # Decode through stages
        for i, (stage, upsample) in enumerate(zip(self.decoder_stages[:-1], self.upsample_layers)):
            x = stage(x)
            x = upsample(x)
        
        # Final stage
        x = self.decoder_stages[-1](x)
        
        # Final upsampling
        x = self.final_upsample(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class ConvNeXtV2MAE(nn.Module):
    '''
    ConvNeXt V2 with Encoder-Decoder structure for MAE training.
    '''
    def __init__(self, in_chans=12, depths=[3,3,9,3], dims=[96,192,384,768], 
                 decoder_dims=[512,256,128,64], mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = 16  # Effective patch size after 4x4 stem + 3 2x2 downsamples
        
        self.encoder = ConvNeXtV2Encoder(in_chans, depths, dims)
        self.decoder = ConvNeXtV2Decoder(dims, decoder_dims, in_chans)
        
        # Mask token (learnable parameter)
        self.mask_token = nn.Parameter(torch.zeros(1, dims[-1], 1, 1))
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
    
    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, C, H, W]
        """
        N, C, H, W = x.shape
        L = H * W  # number of patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.reshape(N, 1, H, W)
        
        return mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Apply masking
        mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply mask to input (set masked patches to 0)
        x_masked = x * (1 - mask)
        
        # Encode
        latent, features = self.encoder(x_masked)
        
        return latent, mask, ids_restore, features
    
    def forward_decoder(self, x, ids_restore=None):
        # Decode
        pred = self.decoder(x)
        return pred
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, 1, H, W], 0 is keep, 1 is remove
        """
        target = imgs
        loss = (pred - target) ** 2
        loss = loss.mean(dim=1, keepdim=True)  # [N, 1, H, W], mean loss per patch
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore, features = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

# For backwards compatibility
class ConvNeXtV2(ConvNeXtV2Encoder):
    '''
    Original ConvNeXt V2 (encoder only) for backwards compatibility.
    '''
    def forward(self, x):
        latent, features = super().forward(x)
        return latent

if __name__ == '__main__':
    # Test the MAE model
    x = torch.rand((2, 12, 256, 256))
    
    # Test encoder-decoder
    model = ConvNeXtV2MAE(in_chans=12)
    loss, pred, mask = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Loss: {loss.item()}")
    
    # Test original model (backwards compatibility)
    original_model = ConvNeXtV2(in_chans=12)
    y = original_model(x)
    print(f"Original model output shape: {y.shape}")