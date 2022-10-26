
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import utils
from .TFA import TFA

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class PhysNet_padding_ED_peak(nn.Module):
    def __init__(self, frames=160 ,device_ids = [0],hidden_layer = 128):
        super(PhysNet_padding_ED_peak, self).__init__()
        imnet_in_dim = 16*9+2
        self.device_ids = device_ids
        self.hidden_layer = hidden_layer        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock1_1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 2, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.upconv = nn.ConvTranspose3d(16, 16, [1, 5, 5], stride=[1,2,2], padding=[0, 2, 2])
        self.mlp_x = MLP(in_dim = imnet_in_dim, out_dim = 16 , hidden_list = [self.hidden_layer,self.hidden_layer])
        self.mlp_y = MLP(in_dim = imnet_in_dim, out_dim = 16 , hidden_list = [self.hidden_layer,self.hidden_layer])
        self.tfa = TFA(num_feat=16, num_block=15, spynet_path='absolute path of spynet_.pth')

    def forward(self, x, y):  # x [3, T, 128,128]
        x_visual = x
        z = x.permute(0,2,1,3,4).contiguous()
        z = self.tfa(z)
        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]


        y = self.ConvBlock1_1(y)
        y = self.MaxpoolSpa(y)
        
        x = self.pfe_heart_x(x,64)
        y = self.pfe_heart_y(y,64)
        x = x+z
        y = y+z
        x = x.permute(0,2,1,3,4).contiguous()
        y = y.permute(0,2,1,3,4).contiguous()
        x = torch.cat([x,y],dim=0)
        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [32, T/2, 32,32]    Temporal half

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        x = self.poolspa(x)  # x [64, T, 1,1]
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG_peak = x.squeeze(-1).squeeze(-1)  # [Batch, 2, T]

        return rPPG_peak, x_visual, x_visual3232, x_visual1616

    def pfe_heart_x(self,x,output_size):

        [batch, channel, length, width, height] = x.shape
        imnet = self.mlp_x
        coord = utils.make_coord([64,64]).cuda(device=self.device_ids[0])

        cell = torch.ones_like(coord)
        coord = coord.expand(batch*length,-1,-1)
        cell[:, 0] *= 2 / output_size
        cell[:, 1] *= 2 / output_size
        cell = cell.expand(batch*length,-1,-1)
        x = x.permute(0,2,1,3,4).contiguous().view(-1,channel,width,height)
        ret = self.query_rgb(x, coord,cell=cell,imnet=imnet)
        ret = ret.permute(0,2,1).contiguous().view(batch*length,channel,output_size,output_size).view(batch,length,channel,output_size,output_size)

        return ret

    def pfe_heart_y(self,x,output_size):
        [batch, channel, length, width, height] = x.shape
        imnet = self.mlp_y
        coord = utils.make_coord([64,64]).cuda(device=self.device_ids[0])

        cell = torch.ones_like(coord)
        coord = coord.expand(batch*length,-1,-1)
        cell[:, 0] *= 2 / output_size
        cell[:, 1] *= 2 / output_size
        cell = cell.expand(batch*length,-1,-1)
        x = x.permute(0,2,1,3,4).contiguous().view(-1,channel,width,height)
        ret = self.query_rgb(x, coord,cell=cell,imnet=imnet)
        ret = ret.permute(0,2,1).contiguous().view(batch*length,channel,output_size,output_size).view(batch,length,channel,output_size,output_size)
        return ret

    def query_rgb(self, x, coord, cell=None, imnet=None):
        feat = x
        imnet = imnet
        feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])# feature unfolding
        coord_ = coord.clone()


        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)


        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([q_feat, rel_cell], dim=-1)

        bs, q = coord.shape[:2]
        pred = imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        return pred


