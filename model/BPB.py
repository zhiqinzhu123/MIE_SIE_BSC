import torch.nn as nn
import torch.nn.functional as F
import torch

class BPB_Module(nn.Module):
    def __init__(self, inChans):
        super(BPB_Module, self).__init__()
        chans = 5 * inChans
        self.d11 = nn.Conv3d(inChans, inChans, kernel_size=1, dilation=1)
        self.d31 = nn.Conv3d(inChans, inChans, kernel_size=3, padding=1 , dilation=1)
        self.d32 = nn.Conv3d(inChans, inChans, kernel_size=3, padding=2 , dilation=2)
        self.d34 = nn.Conv3d(inChans, inChans, kernel_size=3, padding=4 , dilation=4)
        self.d36 = nn.Conv3d(inChans, inChans, kernel_size=3, padding=6 , dilation=6)
        self.d1 = nn.Conv3d(chans, inChans, kernel_size=1)

    def forward(self, x):
        x0 = x
        x11 = self.d11(x)
        x31 = self.d31(x)
        x32 = self.d32(x)
        x34 = self.d34(x)
        x36 = self.d36(x)
        x = torch.cat((x11, x31, x32, x34, x36), dim=1)
        x = self.d1(x)
        x = torch.sigmoid(x)
        out = x * x0
        out = out + x0
        return out