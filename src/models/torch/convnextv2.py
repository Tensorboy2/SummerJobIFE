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

class ConvNeXtV2SegmentationDecoder(nn.Module):
    '''
    ConvNeXt V2 Segmentation Decoder with skip connections (U-Net style).
    '''
    def __init__(self, encoder_dims=[96,192,384,768], decoder_dims=[384,192,96,48], num_classes=21):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.num_classes = num_classes
        
        # Skip connection projections to match decoder dimensions AFTER upsampling
        self.skip_projections = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):  # Skip the deepest feature
            # Project to the target decoder dimension (after upsampling)
            target_dim = decoder_dims[i + 1]  # The dimension after upsampling
            encoder_skip_dim = encoder_dims[-(i+2)]  # Corresponding encoder dimension
            self.skip_projections.append(
                nn.Conv2d(encoder_skip_dim, target_dim, kernel_size=1)
            )
        
        # Initial projection from encoder to decoder
        self.encoder_to_decoder = nn.Conv2d(encoder_dims[-1], decoder_dims[0], kernel_size=1)
        
        # Decoder blocks with upsampling
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            # Upsampling layer
            upsample = nn.ConvTranspose2d(
                decoder_dims[i], decoder_dims[i+1], 
                kernel_size=2, stride=2
            )
            
            # Fusion layer (combines upsampled features with skip connection)
            # Both upsampled and skip features will have decoder_dims[i+1] channels
            fusion_dim = decoder_dims[i+1] * 2  # Skip + upsampled features
            fusion = nn.Sequential(
                nn.Conv2d(fusion_dim, decoder_dims[i+1], kernel_size=3, padding=1),
                LayerNorm2d(decoder_dims[i+1]),
                nn.GELU()
            )
            
            # ConvNeXt blocks for feature refinement
            blocks = nn.Sequential(
                *[ConvNeXtBlock(dim=decoder_dims[i+1]) for _ in range(2)]
            )
            
            self.decoder_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'fusion': fusion,
                'blocks': blocks
            }))
        
        # Final upsampling to match input resolution (4x upsampling for stem)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            LayerNorm2d(decoder_dims[-1]),
            nn.GELU(),
            *[ConvNeXtBlock(dim=decoder_dims[-1]) for _ in range(2)]
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1], kernel_size=3, padding=1),
            LayerNorm2d(decoder_dims[-1]),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dims[-1], num_classes, kernel_size=1)
        )
        
    def forward(self, encoder_features):
        """
        encoder_features: List of features from encoder stages
        """
        # Debug: print encoder features shapes
        # print("Encoder features shapes:", [f.shape for f in encoder_features])
        # print("Decoder dims:", self.decoder_dims)
        
        # Start with the deepest features
        x = self.encoder_to_decoder(encoder_features[-1])
        # print(f"After encoder_to_decoder: {x.shape}")
        
        # Progressive upsampling with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # print(f"\nDecoder block {i}:")
            # print(f"  Input x shape: {x.shape}")
            
            # Upsample current features
            x = decoder_block['upsample'](x)
            # print(f"  After upsample: {x.shape}")
            
            # Get corresponding skip connection (reverse order)
            skip_idx = len(encoder_features) - 2 - i
            skip_features = encoder_features[skip_idx]
            # print(f"  Skip features (idx {skip_idx}) original shape: {skip_features.shape}")
            
            # Project skip features to match the upsampled feature dimension
            skip_features = self.skip_projections[i](skip_features)
            # print(f"  Skip features after projection: {skip_features.shape}")
            
            # Ensure spatial dimensions match before concatenation
            if x.shape[2:] != skip_features.shape[2:]:
                skip_features = F.interpolate(skip_features, size=x.shape[2:], mode='bilinear', align_corners=False)
                # print(f"  Skip features after interpolation: {skip_features.shape}")
            
            # Concatenate upsampled features with skip connection
            x = torch.cat([x, skip_features], dim=1)
            # print(f"  After concatenation: {x.shape}")
            
            # Fuse and refine features
            x = decoder_block['fusion'](x)
            # print(f"  After fusion: {x.shape}")
            x = decoder_block['blocks'](x)
            # print(f"  After blocks: {x.shape}")
        
        # Final upsampling to input resolution
        x = self.final_upsample(x)
        # print(f"After final upsample: {x.shape}")
        
        # Classification
        x = self.classifier(x)
        # print(f"Final output: {x.shape}")
        
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

class ConvNeXtV2Segmentation(nn.Module):
    '''
    ConvNeXt V2 for semantic segmentation with U-Net style architecture.
    '''
    def __init__(self, in_chans=3, num_classes=21, depths=[3,3,9,3], dims=[96,192,384,768], 
                 decoder_dims=[384,192,96,48]):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = ConvNeXtV2Encoder(in_chans, depths, dims)
        self.decoder = ConvNeXtV2SegmentationDecoder(dims, decoder_dims, num_classes)
        
    def forward(self, x):
        # Encode
        _, encoder_features = self.encoder(x)
        
        # Decode with skip connections
        segmentation_logits = self.decoder(encoder_features)
        
        return segmentation_logits

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
    print(f"MAE Input shape: {x.shape}")
    print(f"MAE Prediction shape: {pred.shape}")
    print(f"MAE Mask shape: {mask.shape}")
    print(f"MAE Loss: {loss.item()}")
    
    # Test segmentation model
    x_seg = torch.rand((2, 3, 256, 256))
    seg_model = ConvNeXtV2Segmentation(in_chans=3, num_classes=1)
    seg_output = seg_model(x_seg)
    print(f"\nSegmentation Input shape: {x_seg.shape}")
    print(f"Segmentation Output shape: {seg_output.shape}")
    
    # Test original model (backwards compatibility)
    original_model = ConvNeXtV2(in_chans=12)
    y = original_model(x)
    print(f"\nOriginal model output shape: {y.shape}")
    
    # Test with different input sizes
    x_test = torch.rand((1, 3, 512, 512))
    seg_output_large = seg_model(x_test)
    print(f"\nLarge input shape: {x_test.shape}")
    print(f"Large segmentation output shape: {seg_output_large.shape}")