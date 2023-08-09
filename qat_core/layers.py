###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch, sys
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from qat_core.functions import quantization, clamping_qa, clamping_hw, calc_out_shift, calc_out_shift_rho
import torch.nn.functional as F

###################################################
### Base layer for conv/linear, 
###    enabling quantization-related mechanisms
class shallow_base_layer(nn.Module):
    def __init__(
            self,
            quantization_mode = 'fpt', # 'fpt', 'fpt_unconstrained', 'qat', 'qat_ap' and 'eval'
            pooling_tuple     = False, # boolean flag, none or tuple (kernel,stride,padding)
            # if it is a tuple with (kernel_size, stride, padding) arguments it sets the pooling with these parameters
            # if it is True, then it sets kernel_size = 2, stride = 2, padding = 0
            # if it is False, then it sets the pooling None
            # if it is None, it sets the pooling None
            operation_module  = None,  # torch nn module for keeping and updating conv/linear parameters
            operation_fcnl    = None,  # torch nn.functional for actually doing the operation
            activation_module = None,  # torch nn module for relu/abs
            batchnorm_module  = None,  # torch nn module for batchnorm, see super
            conv_groups       = None,  # we use this to do only depthwise for now. so grouped conv only possible with num_groups=C_in_channels
            output_width_30b  = False, # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
            padding_mode      = "zeros", # used to decide which type of padding operation among "zeros", "reflect", "replicate" and "circular" is to be performed. default with mode "zeros" and padding value 0 corresponds to no padding
            transposed        = False  # either the operation is convtransposed or not
        ):
        super().__init__()

        ###############################################################################
        # Initialize stuff that won't change throughout the model's lifetime here
        # since this place will only be run once (first time the model is declared)
        if isinstance(pooling_tuple, tuple) and (len(pooling_tuple) == 3):
            self.pool   = nn.MaxPool2d(kernel_size=pooling_tuple[0], stride=pooling_tuple[1], padding=pooling_tuple[2])
        elif (type(pooling_tuple) == type(True)):
            if(pooling_tuple==True):
                self.pool   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            else:
                self.pool   = None
        elif pooling_tuple == None:
            self.pool   = None
        else:
            print('wrong pooling type in model. (kernel,stride,padding) as a tuple are acceptable. exiting')
            sys.exit()


        ### Burak: we have to access and change (forward pass) and also train (backward pass) parameters .weight and .bias for the operations
        ###        therefore we keep both a functional and a module for Conv2d/Linear. The name "op" is mandatory for keeping params in Maxim
        ###        experiment format.
        self.op         = operation_module
        self.op_fcn     = operation_fcnl
        self.act        = activation_module
        self.bn         = batchnorm_module
        self.wide       = output_width_30b
        self.dw_groups  = conv_groups
        self.padding_mode     = padding_mode
        self.transposed = transposed

        ###############################################################################
        # Initialize stuff that will change during mode progression (FPT->QAT->Eval/HW).
        self.mode               = quantization_mode;
        self.quantize_Q_ud_8b   = None
        self.quantize_Q_ud_wb   = None
        self.quantize_Q_ud_bb   = None
        self.quantize_Q_ud_ap   = None
        self.quantize_Q_d_8b    = None
        self.quantize_Q_u_wb    = None
        self.quantize_Q_ud_wide = None
        self.quantize_Q_d_wide = None
        self.clamp_C_qa_8b = None
        self.clamp_C_qa_bb = None
        self.clamp_C_qa_wb = None
        self.clamp_C_hw_8b = None
        self.clamp_C_qa_wide = None
        self.clamp_C_hw_wide = None

        ### Burak: these aren't really trainable parameters, but they're logged in the Maxim experiment format. It seems they marked
        ###        them as "non-trainable parameters" to get them automatically saved in the state_dict
        self.output_shift        = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: we called this los, this varies, default:0
        self.weight_bits         = nn.Parameter(torch.Tensor([ 8 ]), requires_grad=False) ### Burak: we called this wb, this varies, default:8
        self.bias_bits           = nn.Parameter(torch.Tensor([ 8 ]), requires_grad=False) ### Burak: this is always 8
        self.quantize_activation = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware, default: fpt
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware, default: fpt
        self.shift_quantile      = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this varies, default:1 (naive)

        ###############################################################################
        # Do first mode progression (to the default)
        ### Burak: this recognizes that layer configuration is done via a function,
        ###        thus, can be done again in training time for mode progression
        weight_bits      = self.weight_bits
        bias_bits        = self.bias_bits
        shift_quantile   = self.shift_quantile
        self.configure_layer_base( weight_bits, bias_bits, shift_quantile )


    # This will be called during mode progression to set fields,
    # check workflow-training-modes.png in doc for further info.
    # sets functions for all modes though, not just the selected mode
    def configure_layer_base(self, weight_bits, bias_bits, shift_quantile):
        # quantization operators
        self.quantize_Q_ud_8b = quantization(xb=8, mode='updown', wide=False, m=None, g=None)  # 8 here is activation bits
        self.quantize_Q_ud_wb = quantization(xb=weight_bits, mode='updown', wide=False, m=None, g=None)
        self.quantize_Q_ud_bb = quantization(xb=bias_bits, mode='updown', wide=False, m=None, g=None)
        self.quantize_Q_ud_ap = quantization(xb=2, mode='updown_ap', wide=False, m=None, g=None)  # 2 here is dummy, mode antipodal overrides xb
        self.quantize_Q_d_8b = quantization(xb=8, mode='down', wide=False, m=None, g=None)  # 8 here is activation bits
        self.quantize_Q_u_wb = quantization(xb=weight_bits, mode='up', wide=False, m=None, g=None)
        self.quantize_Q_ud_wide = quantization(xb=8, mode='updown', wide=True, m=None, g=None)  # 8 here is activation bits, but its wide, so check inside
        self.quantize_Q_d_wide = quantization(xb=8, mode='down', wide=True, m=None, g=None)  # 8 here is activation bits, but its wide, so check inside

        # clamping operators
        self.clamp_C_qa_8b    = clamping_qa(xb = 8,           wide=False) # 8 here is activation bits
        self.clamp_C_qa_bb    = clamping_qa(xb = bias_bits,   wide=False)
        self.clamp_C_qa_wb    = clamping_qa(xb = weight_bits, wide=False)
        self.clamp_C_hw_8b    = clamping_hw(xb = 8,           wide=False) # 8 here is activation bits
        self.clamp_C_qa_wide  = clamping_qa(xb = None,        wide=True)  # None to avoid misleading info on the # of bits, check inside
        self.clamp_C_hw_wide  = clamping_hw(xb = None,        wide=True)  # None to avoid misleading info on the # of bits, check inside

        # state variables
        self.weight_bits = nn.Parameter(torch.Tensor([weight_bits]), requires_grad=False)
        self.bias_bits = nn.Parameter(torch.Tensor([bias_bits]), requires_grad=False)
        self.shift_quantile = nn.Parameter(torch.Tensor([shift_quantile]), requires_grad=False)

    # This will be called during mode progression, during training (scale bn parameters with 4)
    def mode_fptunconstrained2fpt(self, quantization_mode):
        if(self.bn is not None):
            weightsExist = (self.bn.weight != None)
            biasExist = (self.bn.bias != None)

            # BN outputs multiplied by 4 to compensate effect of dividing 4 at forward pass
            if(weightsExist and biasExist):
                self.bn.weight.data = self.bn.weight.data * 4.0
                self.bn.bias.data   = self.bn.bias.data * 4.0
            else:
                # batchnorm affine=False
                self.bn.running_var = self.bn.running_var / 4.0
        else:
            pass
            #print('This layer does not have batchnorm')

        self.mode                = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware


    # This will be called during mode progression, during training
    def mode_fpt2qat(self, quantization_mode):
        # just fold batchnorms
        if(self.bn is not None):
            weightsExist = (self.bn.weight != None)
            biasExist = (self.bn.bias != None)
            if(weightsExist and biasExist):
                # batchnorm affine=True
                w_fp = self.op.weight.data

                scale_factor = self.bn.weight.data
                shift_factor = self.bn.bias.data

                running_mean_mu     = self.bn.running_mean
                running_var         = self.bn.running_var
                running_stdev_sigma = torch.sqrt(running_var + 1e-5)

                w_hat = scale_factor.reshape(-1,1,1,1) * w_fp * (1.0 / (running_stdev_sigma*4.0)).reshape((w_fp.shape[0],) + (1,) * (len(w_fp.shape) - 1))
                self.op.weight.data = w_hat

                if (self.op.bias != None):
                    b_fp = self.op.bias.data
                    b_hat = scale_factor * (b_fp - running_mean_mu)/(running_stdev_sigma*4.0) + shift_factor/4.0
                    self.op.bias.data   = b_hat
                else:
                    b_fp = torch.zeros_like(shift_factor).to(self.op.weight.data.device)
                    b_hat = scale_factor * (b_fp - running_mean_mu)/(running_stdev_sigma*4.0)  + shift_factor/4.0
                    self.op.register_parameter(name='bias', param=nn.Parameter(torch.zeros_like(b_hat), requires_grad=True))
                    self.op.bias.data   = b_hat

                self.bn             = None

            else:
                # batchnorm affine=False
                w_fp = self.op.weight.data

                running_mean_mu     = self.bn.running_mean
                running_var         = self.bn.running_var
                running_stdev_sigma = torch.sqrt(running_var + 1e-5)

                w_hat =  w_fp * (1.0 / (running_stdev_sigma*4.0)).reshape((w_fp.shape[0],) + (1,) * (len(w_fp.shape) - 1))
                self.op.weight.data = w_hat

                if (self.op.bias != None):
                    b_fp = self.op.bias.data
                    b_hat = (b_fp - running_mean_mu)/(running_stdev_sigma*4.0)
                    self.op.bias.data   = b_hat
                else:
                    b_fp = torch.zeros(self.op.weight.data.shape[0]).to(self.op.weight.data.device)
                    b_hat = (b_fp - running_mean_mu)/(running_stdev_sigma*4.0)
                    self.op.register_parameter(name='bias', param=nn.Parameter(torch.zeros_like(b_hat), requires_grad=True))
                    self.op.bias.data   = b_hat

                self.bn             = None
        else:
            pass
            #print('This layer does not have batchnorm')
        self.mode                = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware

    # This will be called during mode progression after training, for eval
    def mode_qat2hw(self, quantization_mode):
        w_hat = self.op.weight.data
        if self.op.bias is not None:
            b_hat = self.op.bias.data
        else:
            b_hat = None

        shift = -self.output_shift.data;
        s_o   = 2**(shift)
        wb    = self.weight_bits.data.cpu().numpy()[0]

        w_clamp = [-2**(wb-1)  , 2**(wb-1)-1 ]
        b_clamp = [-2**(wb+8-2), 2**(wb+8-2)-1] # 8 here is activation bits

        w = w_hat.mul(2**(wb -1)).mul(s_o).add(0.5).floor()
        w = w.clamp(min=w_clamp[0],max=w_clamp[1])
        
        if b_hat is not None:
            b = b_hat.mul(2**(wb -1 + 7)).mul(s_o).add(0.5).floor()
            b = b.clamp(min=b_clamp[0],max=b_clamp[1])
            self.op.bias.data = b
        else:
            self.op.bias = None

        self.op.weight.data      = w
        self.mode                = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([ 1 ]), requires_grad=False) ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware
        self.adjust_output_shift = nn.Parameter(torch.Tensor([ 0 ]), requires_grad=False) ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware

    def mode_qat_ap2hw(self, quantization_mode):
        w_hat = self.op.weight.data
        b_hat = self.op.bias.data

        shift = -self.output_shift.data;
        s_o = 2 ** (shift)
        wb = self.weight_bits.data.cpu().numpy()[0]

        if (wb == 2):
            w = self.quantize_Q_ud_ap(w_hat).mul(2.0)
        else:
            w_clamp = [-2 ** (wb - 1), 2 ** (wb - 1) - 1]
            w = w_hat.mul(2 ** (wb - 1)).mul(s_o).add(0.5).floor()
            w = w.clamp(min=w_clamp[0], max=w_clamp[1])

        b = b_hat.mul(2 ** (wb - 1 + 7)).mul(s_o).add(0.5).floor()

        b_clamp = [-2 ** (wb + 8 - 2), 2 ** (wb + 8 - 2) - 1]  # 8 here is activation bits
        b = b.clamp(min=b_clamp[0], max=b_clamp[1])

        self.op.weight.data = w
        self.op.bias.data = b
        self.mode = quantization_mode;
        self.quantize_activation = nn.Parameter(torch.Tensor([1]),
                                                requires_grad=False)  ### Burak: this is 0 in FPT, 1 in QAT & eval/hardware


        self.adjust_output_shift = nn.Parameter(torch.Tensor([self.output_shift.data]),
                                                requires_grad=False)  ### Burak: this is 1 in FPT & QAT, 0 in eval/hardware

    def forward(self, x):
        if (self.pool is not None):
            x = self.pool(x)

        if (self.mode == 'fpt'):
            # pre-compute stuff
            w_fp = self.op.weight
            b_fp = self.op.bias

            if not isinstance(self, fullyconnected):
                # actual forward pass
                # Deniz: nn.functional.conv's are not supporting padding modes, so had to add this nn.functional.pad manually.
                # Also, default padding mode names are different for nn.func.pad and nn.conv. Related links:
                # https://discuss.pytorch.org/t/torch-nn-functional-conv1d-padding-like-torch-nn-conv1d/119489
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                if self.op.padding_mode == "zeros":
                    self.op.padding_mode = "constant"
                if self.transposed:
                    x = self.op_fcn(x, w_fp, b_fp, self.op.stride, self.op.padding, self.op.output_padding);
                else:
                    if (self.dw_groups is None):
                        # Note that pad=self.op.padding is just a container
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_fp, b_fp,
                                        self.op.stride, 0, self.op.dilation)
                    else:
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_fp, b_fp,
                                        self.op.stride, 0, self.op.dilation, groups=self.dw_groups)
            else:
                x = self.op_fcn(x, w_fp, b_fp, None, None)

            if (self.bn is not None):
                x = self.bn(x)  # make sure var=1 and mean=0
                x = x / 4.0  # since BN is only making sure var=1 and mean=0, 1/4 is to keep everything within [-1,1] w/ hi prob.
            if (self.act is not None):
                x = self.act(x)
            if ((self.wide) and (self.act is None)):
                x = self.clamp_C_qa_wide(x)
            else:
                x = self.clamp_C_qa_8b(x)

            # save stuff (los is deactivated in fpt)
            self.output_shift = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # functional, used in Maxim-friendly experiments
            self.quantize_activation = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # ceremonial, for Maxim-friendly experiments
            self.adjust_output_shift = nn.Parameter(torch.Tensor([1]), requires_grad=False)  # ceremonial, for Maxim-friendly experiments

        elif (self.mode == 'fpt_unconstrained'):
            # pre-compute stuff
            w_fp = self.op.weight
            b_fp = self.op.bias

            if not isinstance(self, fullyconnected):
                # actual forward pass
                # Deniz: nn.functional.conv's are not supporting padding modes, so had to add this nn.functional.pad manually.
                # Also, default padding mode names are different for nn.func.pad and nn.conv. Related links:
                # https://discuss.pytorch.org/t/torch-nn-functional-conv1d-padding-like-torch-nn-conv1d/119489
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                if self.op.padding_mode == "zeros":
                    self.op.padding_mode = "constant"
                if self.transposed:
                    x = self.op_fcn(x, w_fp, b_fp, self.op.stride, self.op.padding, self.op.output_padding);
                else:
                    if (self.dw_groups is None):
                        # Note that pad=self.op.padding is just a container
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_fp, b_fp,
                                        self.op.stride, 0, self.op.dilation)
                    else:
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_fp, b_fp,
                                        self.op.stride, 0, self.op.dilation, groups=self.dw_groups)
            else:
                x = self.op_fcn(x, w_fp, b_fp, None, None)

            if (self.bn is not None):
                x = self.bn(x)  # make sure var=1 and mean=0
            if (self.act is not None):
                x = self.act(x)
            # save stuff (los is deactivated in fpt)
            self.output_shift = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # functional, used in Maxim-friendly experiments
            self.quantize_activation = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # ceremonial, for Maxim-friendly experiments
            self.adjust_output_shift = nn.Parameter(torch.Tensor([1]), requires_grad=False)  # ceremonial, for Maxim-friendly experiments

        elif (self.mode == 'qat'):
            ###############################################################################
            ## ASSUMPTION: batchnorms are already folded before coming here. Check doc,  ##
            ## the parameters with _fp and with _hat are of different magnitude          ##
            ###############################################################################

            # pre-compute stuff
            w_hat = self.op.weight
            b_hat = self.op.bias
            
            if b_hat is not None:
                los = calc_out_shift(w_hat.detach(), b_hat.detach(), self.shift_quantile.detach())
            else:
                los = calc_out_shift(w_hat.detach(), None, self.shift_quantile.detach())

            s_w = 2 ** (-los)
            s_o = 2 ** (los)
            w_hat_q = self.clamp_C_qa_wb(self.quantize_Q_ud_wb(w_hat * s_w));
            
            if b_hat is not None:
                b_hat_q = self.clamp_C_qa_bb(self.quantize_Q_ud_bb(b_hat * s_w));
            else:
                b_hat_q = None

            if not isinstance(self, fullyconnected):
                # actual forward pass
                # Deniz: nn.functional.conv's are not supporting padding modes, so had to add this nn.functional.pad manually.
                # Also, default padding mode names are different for nn.func.pad and nn.conv. Related links:
                # https://discuss.pytorch.org/t/torch-nn-functional-conv1d-padding-like-torch-nn-conv1d/119489
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                if self.op.padding_mode == "zeros":
                    self.op.padding_mode = "constant"
                if self.transposed:
                    x = self.op_fcn(x, w_hat, b_hat, self.op.stride, self.op.padding, self.op.output_padding);
                else:
                    if (self.dw_groups is None):
                        # Note that pad=self.op.padding is just a container
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_hat_q, b_hat_q,
                                        self.op.stride, 0, self.op.dilation)
                    else:
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_hat_q, b_hat_q,
                                        self.op.stride, 0, self.op.dilation, groups=self.dw_groups)
            else:
                x = self.op_fcn(x, w_hat_q, b_hat_q, None, None)

            x = x * s_o
            if (self.act is not None):
                x = self.act(x)
            if ((self.wide) and (self.act is None)):
                x = self.quantize_Q_ud_wide(x)
                x = self.clamp_C_qa_wide(x)
                ### Deniz: This addition is needed for wide layers to work as expected
                x = x / (2 ** (5));
            else:
                x = self.quantize_Q_ud_8b(x)
                x = self.clamp_C_qa_8b(x)

                # save stuff
            self.output_shift = nn.Parameter(torch.Tensor([los]),
                                             requires_grad=False)  # functional, used in Maxim-friendly checkpoints

        elif (self.mode == 'qat_ap'):
            ###############################################################################
            ## ASSUMPTION: batchnorms are already folded before coming here. Check doc,  ##
            ## the parameters with _fp and with _hat are of different magnitude          ##
            ###############################################################################

            # pre-compute stuff
            w_hat = self.op.weight
            b_hat = self.op.bias
            
            if b_hat is not None:
                los = calc_out_shift(w_hat.detach(), b_hat.detach(), self.shift_quantile.detach())
            else:
                los = calc_out_shift(w_hat.detach(), None, self.shift_quantile.detach())
            
            s_w = 2 ** (-los)
            s_o = 2 ** (los)
            ##############################################
            # This is the only difference from qat
            if (self.weight_bits.data == 2):
                w_hat_q = self.quantize_Q_ud_ap(w_hat * s_w);
            else:
                w_hat_q = self.clamp_C_qa_wb(self.quantize_Q_ud_wb(w_hat * s_w));
            ##############################################
            
            if b_hat is not None:
                b_hat_q = self.clamp_C_qa_bb(self.quantize_Q_ud_bb(b_hat * s_w));
            else:
                b_hat_q = None

            if not isinstance(self, fullyconnected):
                # actual forward pass
                # Deniz: nn.functional.conv's are not supporting padding modes, so had to add this nn.functional.pad manually.
                # Also, default padding mode names are different for nn.func.pad and nn.conv. Related links:
                # https://discuss.pytorch.org/t/torch-nn-functional-conv1d-padding-like-torch-nn-conv1d/119489
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                if self.op.padding_mode == "zeros":
                    self.op.padding_mode = "constant"
                if self.transposed:
                    x = self.op_fcn(x, w_hat, b_hat, self.op.stride, self.op.padding, self.op.output_padding);
                else:
                    if (self.dw_groups is None):
                        # Note that pad=self.op.padding is just a container
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_hat_q, b_hat_q,
                                        self.op.stride, 0, self.op.dilation)
                    else:
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_hat_q, b_hat_q,
                                        self.op.stride, 0, self.op.dilation, groups=self.dw_groups)
            else:
                x = self.op_fcn(x, w_hat_q, b_hat_q, None, None)

            x = x * s_o
            if (self.act is not None):
                x = self.act(x)
            if ((self.wide) and (self.act is None)):
                x = self.quantize_Q_ud_wide(x)
                x = self.clamp_C_qa_wide(x)
                x = x / (2 ** (5));
            else:
                x = self.quantize_Q_ud_8b(x)
                x = self.clamp_C_qa_8b(x)

            # save stuff
            self.output_shift = nn.Parameter(torch.Tensor([los]),
                                             requires_grad=False)  # functional, used in Maxim-friendly checkpoints

        elif self.mode == 'eval':
            #####################################################################################
            ## ASSUMPTION: parameters are already converted to HW before coming here.Check doc ##
            #####################################################################################

            # pre-compute stuff
            w = self.op.weight
            b = self.op.bias
            los = self.output_shift
            s_o = 2 ** los
            w_q = self.quantize_Q_u_wb(w);
            if b is not None:
                b_q = self.quantize_Q_u_wb(b);  # yes, wb, not a typo, they need to be on the same scale
            else:
                b_q = None

            if not isinstance(self, fullyconnected):
                # actual forward pass
                # Deniz: nn.functional.conv's are not supporting padding modes, so had to add this nn.functional.pad manually.
                # Also, default padding mode names are different for nn.func.pad and nn.conv. Related links:
                # https://discuss.pytorch.org/t/torch-nn-functional-conv1d-padding-like-torch-nn-conv1d/119489
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                if self.op.padding_mode == "zeros":
                    self.op.padding_mode = "constant"
                if self.transposed:
                    x = self.op_fcn(x, w, b, self.op.stride, self.op.padding, self.op.output_padding);
                else:
                    if (self.dw_groups is None):
                        # Note that pad=self.op.padding is just a container
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_q, b_q,
                                        self.op.stride, 0, self.op.dilation)
                    else:
                        x = self.op_fcn(F.pad(x, pad=self.op.padding, mode=self.op.padding_mode), w_q, b_q,
                                        self.op.stride, 0, self.op.dilation, groups=self.dw_groups)
            else:
                x = self.op_fcn(x, w_q, b_q, None, None)

            x = x * s_o
            if (self.act is not None):
                x = self.act(x)
            if ((self.wide) and (self.act is None)):
                x = self.quantize_Q_d_wide(x)
                x = self.clamp_C_hw_wide(x)
                ### Deniz: This addition is needed for wide layers to work as expected
                x = x / (2 ** (5));
            else:
                x = self.quantize_Q_d_8b(x)
                x = self.clamp_C_hw_8b(x)

            # nothing to save, this was a hardware-emulated evaluation pass
        else:
            print('wrong quantization mode. should have been one of {fpt, qat, eval}. exiting')
            sys.exit()

        return x


class conv(shallow_base_layer):
    def __init__(
            self,
            C_in_channels      = None,    # number of input channels
            D_out_channels     = None,    # number of output channels
            K_kernel_dimension = None,    # square kernel dimension
            padding            = (1,1,1,1),   # (padding_left, padding_right, padding_top, padding_bottom)
            stride             = (1,1),   # controls the stride of the kernel for width and height
            pooling            = False,   # boolean flag, none or tuple (kernel,stride,padding)
            # if it is a tuple with (kernel_size, stride, padding) arguments it sets the pooling with these parameters
            # if it is True, then it sets kernel_size = 2, stride = 2, padding = 0
            # if it is False, then it sets the pooling None
            # if it is None, it sets the pooling None
            batchnorm          = False,   # boolean flag for now, no trainable affine parameters
            batchnorm_affine   = False,   # boolean flag for now, to do/do not make affine batchnorm operation
            batch_momentum     = 0.05,    # momentum parameter for batchnorm
            num_groups         = None,    # we use this to do only depthwise for now. so grouped conv only possible with num_groups=C_in_channels
            activation         = None,    # 'relu' and 'relu6' are the only choices for now
            bias               = True,    # adds a learnable bias to the output. Default: True
            transposed         = False,   # either conv2d or conv2dtranspose
            output_width_30b   = False,   # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
            weight_initialization=None,
            quantization_mode = 'fpt'
    ):
        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        elif(activation == 'relu6'):
            # Clamping limits get scaled in hw mode, but relu6 cannot be scaled that way.
            print('Warning!!! Relu6 activation is selected for a layer, note that it is only supported for fpt unconstrained mode, it causes unexpected behavior in other modes')
            activation_fcn = nn.ReLU6(inplace=True);
        elif(activation == 'sigmoid'):
            activation_fcn = nn.Sigmoid();
        else:
            print('wrong activation type in model. only {relu and relu6} are acceptable. exiting')
            sys.exit()

        ### Burak: only a module is enough for BN since we neither need to access internals in forward pass, nor train anything (affine=False)
        if(batchnorm):
            if(batchnorm_affine):
                batchnorm_mdl  = nn.BatchNorm2d(D_out_channels, eps=1e-05, momentum=batch_momentum, affine=True)
            else:
                batchnorm_mdl  = nn.BatchNorm2d(D_out_channels, eps=1e-05, momentum=batch_momentum, affine=False)
        else:
            batchnorm_mdl  = None;

        '''
        Groups = 1
            This setting is the default setting. Under this setting, all inputs are convolved to all outputs.
        Groups ≠ 1
            Must be an integer such that the number of input channels and the number of output channels are both divisible by this number.  
            A non-default groups value allows us to create multiple paths where each path connects only a subset of input channels to the output channels.
            For details see : https://iksinc.online/2020/05/10/groups-parameter-of-the-convolution-layer/
        '''

        if transposed:
            if(num_groups is not None):
                print('convtranspose function does not accept groups option. exiting')
                sys.exit()
            else:
                operation_mdl  = nn.ConvTranspose2d(C_in_channels, D_out_channels, kernel_size=K_kernel_dimension, stride=stride, padding=padding, bias=bias); # default is group=1
            operation_fcn  = nn.functional.conv_transpose2d
        else:
            if(num_groups is not None):
                operation_mdl  = nn.Conv2d(C_in_channels, D_out_channels, kernel_size=K_kernel_dimension, stride=stride, padding=padding, bias=bias, groups=num_groups);
            else:
                operation_mdl  = nn.Conv2d(C_in_channels, D_out_channels, kernel_size=K_kernel_dimension, stride=stride, padding=padding, bias=bias); # default is group=1
            operation_fcn  = nn.functional.conv2d

        if weight_initialization is not None:
            weight_initialization(operation_mdl.weight)


        super().__init__(
            pooling_tuple      = pooling,
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            batchnorm_module   = batchnorm_mdl,
            conv_groups        = num_groups,
            output_width_30b   = output_width_30b,
            quantization_mode  = quantization_mode,
            transposed         = transposed
        )


def linear_functional(x, weight, bias, _stride, _padding):
    # dummy linear function that has same arguments as conv
    return nn.functional.linear(x, weight, bias)


class fullyconnected(shallow_base_layer):
    def __init__(
            # This must be updated, batch norm and ReLU6 issues
            self,
            in_features        = None,    # number of output features
            out_features       = None,    # number of output features
            pooling            = False,   # boolean flag, none or tuple (kernel,stride,padding)
            # if it is a tuple with (kernel_size, stride, padding) arguments it sets the pooling with these parameters
            # if it is True, then it sets kernel_size = 2, stride = 2, padding = 0
            # if it is False, then it sets the pooling None
            # if it is None, it sets the pooling None
            batchnorm          = False,   # boolean flag for now, no trainable affine parameters
            activation         = None,    # 'relu' is the only choice for now
            output_width_30b   = False,    # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
            quantization_mode  = 'fpt'
    ):
        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        else:
            print('wrong activation type in model. only {relu} is acceptable. exiting')
            sys.exit()

        ### Burak: only a module is enough for BN since we neither need to access internals in forward pass, nor train anything (affine=False)
        if(batchnorm):
            batchnorm_mdl  = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.05, affine=False)
        else:
            batchnorm_mdl  = None;

        operation_mdl  = nn.Linear(in_features, out_features, bias=True);
        operation_fcn  = linear_functional

        super().__init__(
            pooling_tuple       = pooling,
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            batchnorm_module   = batchnorm_mdl,
            output_width_30b   = output_width_30b,
            quantization_mode  = quantization_mode
        )

        # Define dummy arguments to make Linear and conv compatible in shallow_base_layer.
        # the name "op" here refers to op in super, i.e., in base_layer
        self.op.stride = None
        self.op.padding = None

class add_residual(nn.Module):
    def __init__(self, quantization_mode='fpt', activation=None):
        super().__init__()
        self.mode             = quantization_mode;
        self.clamp_C_qa_8b    = clamping_qa( xb = 8, wide=False) # 8 here is activation bits
        self.clamp_C_hw_8b    = clamping_hw( xb = 8, wide=False) # 8 here is activation bits
        if(activation is None):
            self.activation_fcn = nn.Identity();
        elif(activation == 'relu'):
            self.activation_fcn = nn.ReLU(inplace=True);
        elif(activation == 'relu6'):
            # Clamping limits get scaled in hw mode, but relu6 cannot be scaled that way.
            print('Warning!!! Relu6 activation is selected for a layer, note that it is only supported for fpt unconstrained mode, it causes unexpected behavior in other modes')
            self.activation_fcn = nn.ReLU6(inplace=True);
        elif(activation == 'sigmoid'):
            self.activation_fcn = nn.Sigmoid();
        else:
            print('wrong activation type in model. only {relu and relu6 and sigmoid} are acceptable. exiting')
            sys.exit()
    def mode_fptunconstrained2fpt(self, quantization_mode):
        self.mode = 'fpt'

    def mode_fpt2qat(self, quantization_mode):
        self.mode = 'qat'

    def mode_qat2hw(self, quantization_mode):
        self.mode = 'eval'

    def forward(self, x, res):
        x = self.activation_fcn(x+res)
        if(self.mode == 'fpt_unconstrained'):
            pass
        elif(self.mode == 'fpt'):
            x = self.clamp_C_qa_8b(x)
        elif(self.mode == 'qat'):
            x = self.clamp_C_qa_8b(x)
        elif(self.mode == 'eval'):
            x = self.clamp_C_hw_8b(x)
        else:
            print('wrong quantization mode. should have been one of {fpt_unconstrained, fpt, qat, eval}. exiting')
            sys.exit()
        return x


class conv1d(shallow_base_layer):
    def __init__(
            self,
            C_in_channels      = None,    # number of input channels
            D_out_channels     = None,    # number of output channels
            K_kernel_dimension = None,    # kernel size
            padding            = (0,0),   # (padding_left, padding_right)
            stride             = 1,       # stride
            pooling            = False,   # boolean flag, none or tuple (kernel,stride,padding)
            # if it is a tuple with (kernel_size, stride, padding) arguments it sets the pooling with these parameters
            # if it is True, then it sets kernel_size = 2, stride = 2, padding = 0
            # if it is False, then it sets the pooling None
            # if it is None, it sets the pooling None
            batchnorm          = False,   # boolean flag for now, no trainable affine parameters
            batchnorm_affine   = False,   # boolean flag for now, to do/do not make affine batchnorm operation
            num_groups         = None,    # we use this to do only depthwise for now. so grouped conv only possible with num_groups=C_in_channels
            activation         = None,    # 'relu' is the only choice for now
            bias               = True,    # adds a learnable bias to the output. Default: True
            output_width_30b   = False,   # boolean flag that chooses between "bigdata" (32b) and normal (8b) activation modes for MAX78000
            weight_initialization = None,
            quantization_mode = 'fpt',
            dilation           = 1,       # dilation
            padding_mode       = "zeros"  # used to decide which type of padding operation among "zeros", "reflect", "replicate" and "circular" is to be performed. default with mode "zeros" and padding value 0 corresponds to no padding
    ):

        if(activation is None):
            activation_fcn = None;
        elif(activation == 'relu'):
            activation_fcn = nn.ReLU(inplace=True);
        elif(activation == 'relu6'):
            # Clamping limits get scaled in hw mode, but relu6 cannot be scaled that way.
            print('Warning!!! Relu6 activation is selected for a layer, note that it is only supported for fpt unconstrained mode, it causes unexpected behavior in other modes')
            activation_fcn = nn.ReLU6(inplace=True);
        elif(activation == 'sigmoid'):
            self.activation_fcn = nn.Sigmoid();
        else:
            print('wrong activation type in model. only {relu and relu6 and sigmoid} are acceptable. exiting')
            sys.exit()

        if(batchnorm):
            if(batchnorm_affine):
                batchnorm_mdl  = nn.BatchNorm1d(D_out_channels, eps=1e-05, momentum=0.05, affine=True)
            else:
                batchnorm_mdl  = nn.BatchNorm1d(D_out_channels, eps=1e-05, momentum=0.05, affine=False)
        else:
            batchnorm_mdl  = None;

        if(num_groups is not None):
            if(num_groups != C_in_channels):
                print("only num_groups=C_in_channels (i.e., depthwise) is supported for now, exiting")
                sys.exit()
            if(C_in_channels != D_out_channels): # let's not ignore this even though D_out_channels is redundant here
                print('num_in_channels needs to be equal to num_out_channels for depthwise conv layers, exiting')
                sys.exit()
            operation_mdl  = nn.Conv1d(C_in_channels, C_in_channels, kernel_size=K_kernel_dimension, stride=stride, padding=padding, bias=bias, groups=C_in_channels, dilation=dilation, padding_mode=padding_mode);
        else:
            operation_mdl  = nn.Conv1d(C_in_channels, D_out_channels, kernel_size=K_kernel_dimension, stride=stride, padding=padding, bias=bias, dilation=dilation, padding_mode=padding_mode); # default is group=1
        operation_fcn  = nn.functional.conv1d

        if weight_initialization is not None:
            weight_initialization(operation_mdl.weight)

        super().__init__(
            pooling_tuple       = pooling,
            activation_module  = activation_fcn,
            operation_module   = operation_mdl,
            operation_fcnl     = operation_fcn,
            batchnorm_module   = batchnorm_mdl,
            output_width_30b   = output_width_30b,
            quantization_mode  = quantization_mode,
            conv_groups        = num_groups,
            padding_mode       = padding_mode
        )

class concatenate(nn.Module):
    def __init__(self, quantization_mode='fpt',dim=0):
        super().__init__()
        self.dim = dim
        self.mode = quantization_mode;
        self.clamp_C_qa_8b = clamping_qa( xb = 8, wide=False) # 8 here is activation bits
        self.clamp_C_hw_8b = clamping_hw( xb = 8, wide=False) # 8 here is activation bits

    def mode_fptunconstrained2fpt(self, quantization_mode):
        self.mode = 'fpt'

    def mode_fpt2qat(self, quantization_mode):
        self.mode = 'qat'

    def mode_qat2hw(self, quantization_mode):
        self.mode = 'eval'

    def forward(self, x1, x2):
        if(self.mode == 'fpt_unconstrained'):
            x = torch.cat([x1, x2], dim=self.dim)
        elif(self.mode == 'fpt'):
            x = self.clamp_C_qa_8b(torch.cat([x1, x2], dim=self.dim))
        elif(self.mode == 'qat'):
            x = self.clamp_C_qa_8b(torch.cat([x1, x2], dim=self.dim))
        elif(self.mode == 'eval'):
            x = self.clamp_C_hw_8b(torch.cat([x1, x2], dim=self.dim))
        else:
            print('wrong quantization mode. should have been one of {fpt_unconstrained,fpt, qat, eval}. exiting')
            sys.exit()
        return x

# IMPORTANT: Bu kısım şu an quantization yapmıyor, quantized hale getirilmesi gerekiyor.
class Upsample(nn.Module):
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None,
                 recompute_scale_factor=None
        ):


        super().__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor,mode=mode,align_corners=align_corners)

    def forward(self, x):
        x = self.upsample(x)
        return x
