import torch,math
from torch import nn

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)

def get_dct_weights( width, height, channel, fidx_u, fidx_v):
    # width:    width of input
    # height:   height of input
    # channel:  channel of input
    # fidx_u:   horizontal indices of selected fequency
    # fidx_v:   vertical indices of selected fequency

    dct_weights = torch.zero(1, channel, width, height)
    
    c_part = channel // len(fidx_u)
    # split channel for multi-spectal attention

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)

            return dct_weights

class FcaLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FcaLayer, self).__init__()
        self.register_buffer('pre_computed_dct_weight',
                                get_dct_weights(...))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        n, c, _, _ = x.size()
        y = torch.sum(x*self.pre_computed_dct_weight, dim = [2,3])
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)   

if __name__ == "__main__":
    bs, c, z, h, w = 4, 4, 64, 64, 64
    in_tensor = torch.ones(bs, c, z, h, w)

    Fac = FcaLayer(c)
    print("in shape:",in_tensor.shape)
    out_tensor = Fac(in_tensor)
    print("out shape:", out_tensor.shape)