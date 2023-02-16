from importlib.abc import ResourceLoader
from tarfile import REGULAR_TYPES
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from models.mit import mit_b4

#from torchcrf import CRF

class Simplegate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = torch.chunk(x,2,dim=1)
        
        return x1*x2

class SimpleGLP(nn.Module):
    def __init__(self, max_depth=10.0, is_train=True):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        if is_train:            
            ckpt_path = './models/weights'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64

        self.decoder = Decoder(channels_in, channels_out)
    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        
        self.aspp = ASPP()
        
    
    def forward(self, x):                
        conv1, conv2, conv3, conv4 = self.encoder(x)
        #print("conv 1 to 4", conv1.shape, conv2.shape, conv3.shape, conv4.shape)
        conv4 = self.aspp(conv4)
        out = self.decoder(conv1, conv2, conv3, conv4)
        glp_depth = self.last_layer_depth(out)
        simple_depth = torch.sigmoid(glp_depth) * self.max_depth
        
        
        return {'pred_d': simple_depth}

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)
        #print("x1 to out", x_1.shape, x_2.shape, x_3.shape, out.shape)
        return out

class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2),
                      out_channels=in_channel*2, kernel_size=3, stride=1, padding=1),
            Simplegate())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel), kernel_size=3, stride=1, padding=1),
            Simplegate())

        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 2), 
                               out_channels=2, kernel_size=3, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + \
              x_global * attn[:, 1, :, :].unsqueeze(1)

        return out

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_channel=512, dilations=[6,12,18], out_channels=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_channel, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_channel, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                nn.Conv2d(inplanes, mid_channel, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(mid_channel), nn.ReLU())
        out_channels = out_channels if out_channels is not None else mid_channel
        self.out_conv = nn.Sequential(nn.Conv2d(mid_channel * (2 + len(dilations)), out_channels, 1, bias=False),
                                      nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_channel * (2 + len(dilations)), out_channels, 1, bias=False)

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)


