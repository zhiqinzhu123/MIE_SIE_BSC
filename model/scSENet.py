import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1x1 = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1x1(U)  # U:[bs,c,z,h,w] to q:[bs,1,z,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.Conv_Squeeze = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv3d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, z, h, w] to [bs, c, 1, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

if __name__ == "__main__":
    bs, c, z, h, w = 4, 4, 64, 64, 64
    in_tensor = torch.ones(bs, c, z, h, w)

    sc_se = scSE(c)
    print("in shape:",in_tensor.shape)
    out_tensor = sc_se(in_tensor)
    print("out shape:", out_tensor.shape)