import torch.nn.functional as F
import torch.nn as nn
from ..model.unet import unet3D_m1
from ..model.block import mergeCrop
from ..quant.quant_core import get_quantize_layer
class unet3D_m1_quant(nn.Module): # deployed model-1
    def __init__(self, net, quant_bits=8, rescale_skip='', overflow_rate=0.0, counter=10):
        super(unet3D_m1_quant, self).__init__()
        self.rescale_skip = rescale_skip
        self.net = net
        if self.rescale_skip != '': # need quantization layer after mergeCrop
            self.reS = nn.ModuleList([get_quantize_layer(self.rescale_skip, 'mc'+str(x), quant_bits, overflow_rate, counter)
            for x in range(self.net.depth)])

    def set_rescale_skip_scale(self):
        if self.rescale_skip != '':
            for l in range(self.net.depth):
                keys_down = self.net.downC[self.net.depth-l-1]._modules.keys()
                keys_up = self.net.upS[l]._modules.keys()
                self.reS[l].sf = min(self.net.downC[self.net.depth-l-1]._modules[keys_down[-2]].sf, self.net.upS[l]._modules[keys_up[-1]].sf)
                print l,self.net.downC[self.net.depth-l-1]._modules[keys_down[-2]].sf, self.net.upS[l]._modules[keys_up[-1]].sf

    def forward(self, x):
        if self.rescale_skip == '':
            return self.net.forward(x)
        else:
            down_u = [None]*self.net.depth
            for i in range(self.net.depth):
                down_u[i] = self.net.downC[i](x)
                x = self.net.downS[i](down_u[i])
            x = self.net.center(x)
            for i in range(self.net.depth):
                x = mergeCrop(down_u[self.net.depth-1-i], self.net.upS[i](x))
                x = self.net.upC[i](self.reS[i](x))
            return F.sigmoid(self.net.final(x))

