
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # [B, C, H, W] -> [B, H, W, C] for LayerNorm/Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class Decoder(nn.Module):
    def __init__(self, encoder_output_channels, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(encoder_output_channels, 256, kernel_size=1)
        self.block1 = ConvNeXtBlock(256)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.block2 = ConvNeXtBlock(128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.block3 = ConvNeXtBlock(64)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.block4 = ConvNeXtBlock(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming normal for conv layers, zero for classifier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.conv4(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

    
class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, mmearth_model):
        super().__init__()
        self.initial_conv = mmearth_model.initial_conv
        self.stem = mmearth_model.stem
        self.downsample_layers = mmearth_model.downsample_layers
        self.stages = mmearth_model.stages
        # self.norm = LayerNorm2d()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stem(x)

        x = self.stages[0](x)
        x = self.downsample_layers[0](x)

        x = self.stages[1](x)
        x = self.downsample_layers[1](x)

        x = self.stages[2](x)
        x = self.downsample_layers[2](x)

        x = self.stages[3](x)
        # x = self.norm(x)
        return x

class ConvNeXtV2Segmentation(nn.Module):
    def __init__(self, 
                 in_chans=12, 
                 num_classes=1, encoder_output_channels=320,
                 open_model=False):
        super().__init__()
        mmearth_model = torch.hub.load('vishalned/mmearth-train', 'MPMAE', model_name='convnextv2_atto', pretrained=True, linear_probe=True)
        
        encoder = ConvNeXtV2Encoder(mmearth_model)
        self.encoder = encoder
        self.decoder = Decoder(encoder_output_channels=encoder_output_channels)
        self.num_classes = num_classes

        # Freeze encoder weights
        if open_model:
            self.model_name = 'convnextv2_atto'
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            self.model_name = 'convnextv2_atto'
            for param in self.encoder.parameters():
                param.requires_grad = False
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    model = ConvNeXtV2Segmentation(in_chans=12, num_classes=1, encoder_output_channels=320, open_model=True)
    x = torch.randn(1, 12, 256, 256)
    output = model(x)
    print(output.shape)  # Should be [1, 1, 256, 256]
    print(model.model_name)  # Should print 'convnextv2_open' or 'conv

    state_dict = model.state_dict()
    load = torch.load('results/convnextv2_locked_best_unet_model.pth', map_location='cpu')
    for state_dict,load in zip(state_dict.keys(),load.keys()):
        if state_dict != load:
            print(f"Mismatch: {state_dict} vs {load}")
    
