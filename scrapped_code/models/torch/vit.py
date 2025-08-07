import torch
from torch import nn
import torch.nn.functional as F

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Pre-norm architecture
        normed_x = self.norm1(x)
        attn_output, _ = self.attention(normed_x, normed_x, normed_x)
        x = x + attn_output
        
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output
        return x

class ViTStem(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super(ViTStem, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.conv(x)  # (batch_size, embed_dim, H/patch_size, W/patch_size)
        batch_size, embed_dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        return x, (h, w)  # Return spatial dimensions for decoder

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, mlp_dim, num_layers, patch_size=16, dropout=0.1):
        super(ViTEncoder, self).__init__()
        self.stem = ViTStem(in_channels, embed_dim, patch_size)
        self.layers = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, spatial_dims = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x, spatial_dims

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, patch_size=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
        super(ViTDecoder, self).__init__()
        self.patch_size = patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.num_classes = num_classes
        
        # Project encoder features to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token for missing patches (useful for MAE-style training)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embeddings for decoder
        # Use a large enough number of positions for most reasonable image sizes (e.g., 32x32 patches = 1024)
        self.max_num_patches = 1024
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_num_patches, decoder_embed_dim))
        
        # Transformer blocks in decoder
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_embed_dim, decoder_num_heads, decoder_embed_dim * 4, dropout=0.1) 
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Final prediction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_classes * patch_size * patch_size, bias=True)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, spatial_dims, original_size, ids_restore=None):
        # x shape: (batch_size, num_patches, embed_dim) - encoded features
        # spatial_dims: (height_patches, width_patches)
        # ids_restore: for MAE reconstruction, None for regular segmentation
        
        # Embed tokens to decoder dimension
        x = self.decoder_embed(x)
        
        # Handle MAE case where we need to add mask tokens
        if ids_restore is not None:
            # Add mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Add positional embeddings (always interpolate to match number of patches)
        num_patches = x.shape[1]
        pos_embed = self.decoder_pos_embed[:, :self.max_num_patches, :]
        if num_patches != pos_embed.shape[1]:
            # Interpolate positional embeddings to match the number of patches
            pos_embed = pos_embed.transpose(1, 2)  # (1, C, N)
            pos_embed = F.interpolate(pos_embed, size=num_patches, mode='linear', align_corners=False)
            pos_embed = pos_embed.transpose(1, 2)  # (1, N, C)
        x = x + pos_embed[:, :num_patches, :]
        
        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixels/classes
        x = self.decoder_pred(x)  # (batch_size, num_patches, num_classes * patch_size^2)
        
        # Reshape for spatial output
        batch_size, num_patches, _ = x.shape
        h_patches, w_patches = spatial_dims
        
        # Reshape to get pixel-level predictions
        x = x.view(batch_size, h_patches, w_patches, self.num_classes, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (batch_size, num_classes, h_patches, patch_size, w_patches, patch_size)
        x = x.contiguous().view(batch_size, self.num_classes, h_patches * self.patch_size, w_patches * self.patch_size)
        
        # Resize to original input size if needed
        if original_size is not None:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x

class ViTSegmentation(nn.Module):
    def __init__(self, in_channels=12, embed_dim=768, num_heads=12, mlp_dim=3072, 
                 num_layers=12, num_classes=1, patch_size=16, dropout=0.1,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
        super(ViTSegmentation, self).__init__()
        self.encoder = ViTEncoder(in_channels, embed_dim, num_heads, mlp_dim, num_layers, patch_size, dropout)
        self.decoder = ViTDecoder(embed_dim, num_classes, patch_size, decoder_embed_dim, decoder_depth, decoder_num_heads)

    def forward(self, x):
        original_size = x.shape[-2:]  # (height, width)
        encoded_features, spatial_dims = self.encoder(x)
        segmentation_map = self.decoder(encoded_features, spatial_dims, original_size)
        # Ensure output is (batch, 1, H, W) for binary segmentation
        if segmentation_map.dim() == 3:
            segmentation_map = segmentation_map.unsqueeze(1)
        elif segmentation_map.shape[1] != 1:
            # If num_classes > 1, leave as is; else squeeze to (batch, 1, H, W)
            segmentation_map = segmentation_map[:, :1, ...]
        return segmentation_map

class ViTMAE(nn.Module):
    """Vision Transformer for Masked Autoencoder pretraining"""
    def __init__(self, in_channels=12, embed_dim=768, num_heads=12, mlp_dim=3072, 
                 num_layers=12, patch_size=16, dropout=0.1, mask_ratio=0.75,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
        super(ViTMAE, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = ViTEncoder(in_channels, embed_dim, num_heads, mlp_dim, num_layers, patch_size, dropout)
        
        # Decoder - using the same transformer-based decoder
        self.decoder = ViTDecoder(embed_dim, in_channels, patch_size, decoder_embed_dim, decoder_depth, decoder_num_heads)

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        return x

    def unpatchify(self, x, channels):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Random masking of patches"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x):
        original_size = x.shape[-2:]  # (height, width)
        # Encode
        encoded_features, spatial_dims = self.encoder(x)
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(encoded_features, self.mask_ratio)
        # Decode with transformer blocks
        pred = self.decoder(x_masked, spatial_dims, original_size, ids_restore)
        # Compute loss (for compatibility with ConvNeXtV2MAE and train_mae.py)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    def forward_loss(self, imgs, pred, mask):
        """Calculate reconstruction loss"""
        target = self.patchify(imgs)
        pred_patches = self.patchify(pred)
        
        loss = (pred_patches - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

def create_vit_segmentation(size='base', in_channels=12, num_classes=1, patch_size=16, 
                           decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
    """Create a Vision Transformer model for segmentation tasks"""
    configs = {
        'base': {'embed_dim': 768, 'num_heads': 12, 'mlp_dim': 3072, 'num_layers': 12},
        'large': {'embed_dim': 1024, 'num_heads': 16, 'mlp_dim': 4096, 'num_layers': 24},
        'small': {'embed_dim': 384, 'num_heads': 6, 'mlp_dim': 1536, 'num_layers': 12}
    }
    
    if size not in configs:
        raise ValueError(f"Unsupported ViT size: {size}. Choose from {list(configs.keys())}")
    
    config = configs[size]
    return ViTSegmentation(
        in_channels=in_channels,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        mlp_dim=config['mlp_dim'],
        num_layers=config['num_layers'],
        num_classes=num_classes,
        patch_size=patch_size,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads
    )

def create_vit_mae(size='base', in_channels=12, patch_size=16, mask_ratio=0.75,
                  decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
    """Create a Vision Transformer model for MAE pretraining"""
    configs = {
        'base': {'embed_dim': 768, 'num_heads': 12, 'mlp_dim': 3072, 'num_layers': 12},
        'large': {'embed_dim': 1024, 'num_heads': 16, 'mlp_dim': 4096, 'num_layers': 24},
        'small': {'embed_dim': 384, 'num_heads': 6, 'mlp_dim': 1536, 'num_layers': 12}
    }
    
    if size not in configs:
        raise ValueError(f"Unsupported ViT size: {size}. Choose from {list(configs.keys())}")
    
    config = configs[size]
    return ViTMAE(
        in_channels=in_channels,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        mlp_dim=config['mlp_dim'],
        num_layers=config['num_layers'],
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads
    )

if __name__ == "__main__":
    # Test segmentation model
    # print("Testing ViT Segmentation Model:")
    # seg_model = create_vit_segmentation(size='base', in_channels=12, num_classes=1, patch_size=16)
    
    # # Example input tensor (batch_size, channels, height, width)
    # input_tensor = torch.randn(2, 12, 256, 256)
    # seg_output = seg_model(input_tensor)
    # print(f"Input shape: {input_tensor.shape}")
    # print(f"Segmentation output shape: {seg_output.shape}")  # Should be (2, 1, 256, 256)
    
    # # Test MAE model
    # print("\nTesting ViT MAE Model:")
    # mae_model = create_vit_mae(size='base', in_channels=12, patch_size=16, mask_ratio=0.75)
    
    # loss, mae_output, mask = mae_model(input_tensor)
    # print(f"MAE reconstruction shape: {mae_output.shape}")  # Should be (2, 12, 256, 256)
    # print(f"Mask shape: {mask.shape}")  # Should be (2, 196)
    
    # # Test MAE loss calculation
    # loss = mae_model.forward_loss(input_tensor, mae_output, mask)
    # print(f"MAE reconstruction loss: {loss.item():.4f}")
    
    # # Calculate number of parameters
    # seg_params = sum(p.numel() for p in seg_model.parameters())
    # mae_params = sum(p.numel() for p in mae_model.parameters())
    # print(f"\nSegmentation model parameters: {seg_params:,}")
    # print(f"MAE model parameters: {mae_params:,}")
    
    # # Test with different decoder configurations
    # print("\nTesting with lighter decoder:")
    # light_seg_model = create_vit_segmentation(
    #     size='base', in_channels=12, num_classes=1, patch_size=16,
    #     decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8
    # )
    # light_params = sum(p.numel() for p in light_seg_model.parameters())
    # print(f"Light segmentation model parameters: {light_params:,}")
    # light_output = light_seg_model(input_tensor)
    # print(f"Light model output shape: {light_output.shape}")  # Should be (2, 1, 224, 224)

    # --- Transfer MAE encoder weights to segmentation encoder ---
    print("\nTesting encoder weight transfer from MAE to Segmentation model:")
    input_tensor = torch.randn(2, 12, 256, 256)

    # Create fresh models
    size = 'small'  # or 'large', 'small'
    mae_model = create_vit_mae(size=size, in_channels=12, patch_size=16, mask_ratio=0.75)
    seg_model = create_vit_segmentation(size=size, in_channels=12, num_classes=1, patch_size=16)
    # Get encoder state dict from MAE
    mae_encoder_state = mae_model.encoder.state_dict()
    # Load into segmentation encoder
    missing, unexpected = seg_model.encoder.load_state_dict(mae_encoder_state, strict=True)
    print(f"Loaded MAE encoder state dict into segmentation encoder.")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    # Test segmentation model after weight transfer
    seg_output = seg_model(input_tensor)
    print(f"Segmentation output shape after encoder weight transfer: {seg_output.shape}")