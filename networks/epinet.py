import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_planes, out_planes, kernel_size=2, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

def layer1_multistream(filt_num, view_n):
    return nn.Sequential(
        conv(view_n,filt_num),
        conv(filt_num,filt_num, batchNorm = True),
        conv(filt_num, filt_num),
        conv(filt_num,filt_num, batchNorm = True),
        conv(filt_num, filt_num),
        conv(filt_num,filt_num, batchNorm = True)       
        )

def layer2_merged(filt_num,conv_depth):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    layers = []
    for _ in range(conv_depth):
        layers.append(conv(filt_num,filt_num))
        layers.append(conv(filt_num,filt_num, batchNorm = True))

    return nn.Sequential(*layers)

def layer3_last(filt_num):
    return nn.Sequential(conv(filt_num,filt_num), 
                         nn.Conv2d(filt_num, 1, 2)
    )

class EPINet(nn.Module):
    def __init__(self, filt_num, view_n, conv_depth):
        super(EPINet,self).__init__()
        self.layer1_multistream_90d = layer1_multistream(filt_num, view_n)
        self.layer1_multistream_0d = layer1_multistream(filt_num, view_n)
        self.layer1_multistream_45d = layer1_multistream(filt_num, view_n)        
        self.layer1_multistream_M45d = layer1_multistream(filt_num, view_n)

        self.layer2_merged = layer2_merged(4*filt_num, conv_depth)

        self.layer3_last = layer3_last(4*filt_num)

    def forward(self, input_stack_90d, input_stack_0d, input_stack_45d, input_stack_M45d):
        mid_90d = self.layer1_multistream_90d(input_stack_90d)
        mid_0d = self.layer1_multistream_0d(input_stack_0d)
        mid_45d = self.layer1_multistream_45d(input_stack_45d)
        mid_M45d = self.layer1_multistream_M45d(input_stack_M45d)

        mid_merged = torch.cat((mid_90d,mid_0d,mid_45d,mid_M45d),1) 

        mid_merged_ = self.layer2_merged(mid_merged)

        out = self.layer3_last(mid_merged_)

        return out

        
if __name__ == "__main__":

    x = torch.zeros(2,9,25,25).cuda()
    model = EPINet(70,9,7).cuda()
    print(model(x,x,x,x).shape)
    from torchsummary import summary
    summary(model, [(9,25,25),(9,25,25),(9,25,25),(9,25,25)])

    