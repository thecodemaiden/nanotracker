###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
from abc import ABC

import torch, sys
import torch.nn as nn
from torch.autograd import Function

###################################################
### Quantization Functions
###     backward passes are straight through

## Up-Down (ud) quantization for wide last layer ("bigdata"). Used in QAT
class Q_ud_wide(Function):
    @staticmethod
    def forward(_, x, xb, extrab):
        up_factor = 2 ** (xb - extrab - 1)
        down_factor = 2 ** (xb - 1)
        return x.mul(up_factor).add(.5).floor().div(down_factor)

    @staticmethod
    def backward(_, x):
        return x, None, None


## Up-Down (ud) quantization. Used in QAT
class Q_ud(Function):
    @staticmethod
    def forward(_, x, xb):
        updown_factor = 2 ** (xb - 1)
        return x.mul(updown_factor).add(.5).floor().div(updown_factor)

    @staticmethod
    def backward(_, x):
        return x, None


## Up-Down (ud) quantization for antipodal binary. Used in qat-ap
class Q_ud_ap(Function):
    @staticmethod
    def forward(_, x):
        x = torch.sign(x).div(2.0)  # antipodal (-1,+1) weights @HW correspond to (-0.5,+0.5) in qat
        mask = (x == 0)
        return x - mask.float().div(2.0)

    @staticmethod
    def backward(_, x):
        return x


## Up (u) quantization. Used in Eval/hardware
class Q_u(Function):
    @staticmethod
    def forward(_, x, xb):
        up_factor = 2 ** (8 - xb)
        return x.mul(up_factor).add(.5).floor()  ### Burak: maxim has a .add(0.5) at the beginning, I think that's wrong

    @staticmethod
    def backward(_, x):
        return x, None


## Down (d) quantization. Used in Eval/hardware
class Q_d(Function):
    @staticmethod
    def forward(_, x, xb):
        down_factor = 2 ** (xb-1)
        return x.div(down_factor).add(.5).floor()  ### Burak: maxim has a .add(0.5) at the beginning, I think that's wrong

    @staticmethod
    def backward(_, x):
        return x, None

###################################################
### Quantization module 
###     ("umbrella" for Functions)
class quantization(nn.Module):
    def __init__(self, xb=8, mode='updown', wide=False, m=None, g=None):
        super().__init__()
        self.xb = xb
        self.mode = mode
        self.wide = wide
        self.m = m
        self.g = g

    def forward(self, x):
        ### Deniz: Wide mode was not functioning as expected, so changed with the code from older repo
        if(self.mode=='updown'):
            if(self.wide):
                ### Burak: maxim's implementation had the third argument as +1, which was wrong. 
                ###        the chip seems to be adding 5 more bits to the fractional part
                return Q_ud_wide.apply(x, self.xb, -5)
            else:
                return Q_ud.apply(x, self.xb)
        elif(self.mode=='down'):
            if(self.wide):
                ### Burak: maxim's implementation had the second argument as (self.xb + 1), which was wrong. 
                ###        the chip seems to be adding 5 more bits to the fractional part
                return Q_d.apply(x, self.xb - 5)
            else:
                return Q_d.apply(x, self.xb)
        elif (self.mode == 'up'):
            return Q_u.apply(x, self.xb)
        elif (self.mode == 'updown_ap'):
            return Q_ud_ap.apply(x)
        else:
            print('wrong quantization mode. exiting')
            sys.exit()


###################################################
### Clamping modules
### (doesn't need Functions since backward passes are well-defined)
class clamping_qa(nn.Module):
    def __init__(self, xb=8, wide=False):
        super().__init__()
        if (wide):
            self.min_val = -16384.0  ### Burak: this is wrong, but it's how maxim currently does it, so we play along
            self.max_val = 16383.0  ### Burak: this is wrong, but it's how maxim currently does it, so we play along
        else:
            self.min_val = -1.0
            self.max_val = (2 ** (xb - 1) - 1) / (2 ** (xb - 1))

    def forward(self, x):
        return x.clamp(min=self.min_val, max=self.max_val)


class clamping_hw(nn.Module):
    def __init__(self, xb=8, wide=False):
        super().__init__()
        if(wide):
            self.min_val = -2 ** (30-1)   ### Burak: this is wrong, but it's how maxim currently does it, so we play along
            self.max_val =  2 ** (30-1)-1 ### Burak: this is wrong, but it's how maxim currently does it, so we play along
        else:
            self.min_val = -2 ** (xb - 1)
            self.max_val = 2 ** (xb - 1) - 1

    def forward(self, x):
        return x.clamp(min=self.min_val, max=self.max_val)


###################################################
### Computing output_shift, i.e., "los"
def calc_out_shift(weight, bias, shift_quantile):
    weight_r = torch.flatten(weight)
    if bias is not None:
        bias_r = torch.flatten(bias)
        params_r = torch.cat((weight_r, bias_r))
    else:
        params_r = weight_r
    limit = torch.quantile(params_r.abs(), shift_quantile)
    return -(1. / limit).log2().floor().clamp(min=-15., max=15.)

def calc_out_shift_rho(W):
    # eqn. 22 in the AHA report

    # this code segment is taken from the v1 repo, duygu-yavuz-dev branch
    # layers.py shift_amount_1bit function
    limit = torch.quantile(W.abs(), 1.0)
    return - (1. / limit).log2().ceil().clamp(min=-15., max=15.)
