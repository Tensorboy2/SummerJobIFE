import torch 
import torch.nn as nn

class Unet(nn.Module):
    '''
    Simple U-net for segmentation of FPVs.
    '''
    def __init__(self):
        super.__init__(self)
