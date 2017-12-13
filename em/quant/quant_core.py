from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math
from numpy import log2

from IPython import embed

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()[0]
    sf = math.ceil(log2(v+1e-12))
    return sf

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits)
    v = torch.exp(v) * s
    del input0 # save memory
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits)
    v = torch.exp(v) * s
    del input0 # save memory
    return v

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy()[0])
        min_val = float(min_val.data.cpu().numpy()[0])

    if max_val == min_val:
        v = input # otherwise nan..
    else:
        input_rescale = (input - min_val) / (max_val - min_val)
        n = math.pow(2.0, bits) - 1
        v = torch.floor(input_rescale * n + 0.5) / n
        v =  v * (max_val - min_val) + min_val
        del input_rescale # save memory
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    del input_rescale # save memory
    return v


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)

def quantize_weight(state_dict, bits=8, do_bias=True, overflow_rate=0.0, quant_method='linear'):
    state_dict_quant = OrderedDict()
    for k, v in state_dict.items():
        if not do_bias and '.bias' in k:
            state_dict_quant[k] = v
            continue

        if quant_method == 'linear':
            sf = bits - 1. - compute_integral_part(v, overflow_rate=overflow_rate)
            v_quant  = linear_quantize(v, sf, bits=bits)
        elif quant_method == 'log':
            v_quant = log_minmax_quantize(v, bits=bits)
        elif quant_method == 'minmax':
            v_quant = min_max_quantize(v, bits=bits)
        else:
            v_quant = tanh_quantize(v, bits=bits)
        state_dict_quant[k] = v_quant
    return state_dict_quant

def quantize_feat(model, bits=8, overflow_rate=0.0, quant_method='linear', counter=10):
    assert quant_method in ['linear', 'minmax', 'log', 'tanh']
    for seq_id in range(model.seq_num):
        seq = model.getLearnableSeq(seq_id)
        l = OrderedDict()
        for k, v in seq._modules.items():
            l[k] = v
            if isinstance(v, (nn.Conv3d, nn.Linear, nn.BatchNorm3d, nn.AvgPool3d)):
                if quant_method == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif quant_method == 'log':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif quant_method == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                elif quant_method == 'tanh':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, quant_method)] = quant_layer
        model.setLearnableSeq(seq_id, nn.Sequential(l))
