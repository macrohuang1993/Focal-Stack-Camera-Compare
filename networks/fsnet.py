import torch
import torch.nn as nn
class FSNet(nn.Module):
    """
    Network for disparity field estimation from FS, using dialated Convolution.
    Input:
        x:FS of shape B,nF,C,H,W
        disp_mult: uplimit of the abs of the disparity. To be constrained by tanh
        offset: offset of tanh in output.
    Output:
        disparity of shape B,1,H,W
    """
    def __init__(self,nF, disp_mult, offset):
        super(FSNet,self).__init__()
        self.disp_mult = disp_mult
        self.offset = offset
        C = 3*nF
        self.cnn_layers = nn.Sequential(
            cnn_layer(C,16),#conv => BN => LeakyReLU
            cnn_layer(16,64),
            cnn_layer(64,128),
            cnn_layer(128,128,dilation_rate = 2),
            cnn_layer(128,128,dilation_rate = 4),
            cnn_layer(128,128,dilation_rate = 8),
            cnn_layer(128,128,dilation_rate = 16),
            cnn_layer(128,128),
            cnn_layer(128,128),
            cnn_layer_plain(128, 1)
        )
        self.tanh_NL = nn.Tanh()
    def forward(self,x):
        B,nF,C,H,W = x.shape #input x is the FS
        x = x.reshape(B,C*nF,H,W)
        x = self.cnn_layers(x) # A series of convolution (some dilated)
        x = self.disp_mult * self.tanh_NL(x) + self.offset #constrain the output range
        return x #Estimated depth fields, B,1, H,W


class FSNet_two_resolution_v1(nn.Module):
    """
    Network for disparity field estimation from two_resolution_FS, using dialated Convolution.
    Input:
        x:FS of type list [FS_back_plane and FS_remaining] of shape 1xCx2*Hx2*W and nF-1xCxHxW
        disp_mult: uplimit of the abs of the disparity. To be constrained by tanh
        offset: offset of tanh in output.
    Output:
        disparity of shape B,1,H,W
    """
    def __init__(self,nF, disp_mult, offset):
        super(FSNet_two_resolution_v1,self).__init__()
        self.disp_mult = disp_mult
        self.offset = offset
        C = 3*nF
        self.downsample_layers = nn.Sequential(cnn_layer(3,3),cnn_layer(3,3),nn.Conv2d(3,3,3,2,1), nn.LeakyReLU(inplace=True)) # 3 layers of conv with last layer stride 2 to downsample
        self.cnn_layers = nn.Sequential(
            cnn_layer(C,16),#conv => BN => LeakyReLU
            cnn_layer(16,64),
            cnn_layer(64,128),
            cnn_layer(128,128,dilation_rate = 2),
            cnn_layer(128,128,dilation_rate = 4),
            cnn_layer(128,128,dilation_rate = 8),
            cnn_layer(128,128,dilation_rate = 16),
            cnn_layer(128,128),
            cnn_layer(128,128),
            cnn_layer_plain(128, 1)
        )
        self.tanh_NL = nn.Tanh()
    def forward(self,x):
        FS_back_plane, FS_remaining = x[0], x[1]
        B,nFm1,C,H,W = FS_remaining.shape #input x is the FS
        nF = nFm1 + 1
        FS_remaining = FS_remaining.reshape(B,C*nFm1,H,W)
        FS_back_plane = FS_back_plane.reshape(B,C,2*H,2*W)
        FS_back_plane_feature = self.downsample_layers(FS_back_plane)
        x = torch.cat([FS_back_plane_feature, FS_remaining], dim=1)
        
        x = self.cnn_layers(x) # A series of convolution (some dilated)
        x = self.disp_mult * self.tanh_NL(x) + self.offset #constrain the output range
        return x #Estimated depth fields, B,1, H,W
        
class cnn_layer(nn.Module):
    '''((possibly dilated)conv => BN => LeakyReLU), following learning Local_light field synthesis paper, used in depth_network_pt'''
    def __init__(self, in_ch, out_ch,filter_size = 3, dilation_rate = 1):
        super(cnn_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(dilation_rate*(filter_size-1)//2),
            nn.Conv2d(in_ch, out_ch, filter_size, padding=0,dilation = dilation_rate),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class cnn_layer_plain(nn.Module):
    '''((possibly dilated)conv), following learning Local_light field synthesis paper, used in depth_network_pt'''

    def __init__(self, in_ch, out_ch,filter_size = 3, dilation_rate = 1):
        super(cnn_layer_plain, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(dilation_rate*(filter_size-1)//2),
            nn.Conv2d(in_ch, out_ch, filter_size, padding=0,dilation = dilation_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

        
if __name__ == "__main__":

    x = torch.zeros(2,4,3,25,25).cuda()
    model = FSNet(4,2).cuda()
    print(model(x).shape)
