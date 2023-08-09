###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. main@shallow.ai #
###########################################################################
import torch.nn as nn
import qat_core.layers as layers

class mnv2_SSDlite(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(mnv2_SSDlite, self).__init__()

        self.conv1 = layers.conv(C_in_channels=in_channels, D_out_channels=32, K_kernel_dimension=3, stride=2, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')

        self.epw_conv2 = layers.conv(C_in_channels=32,    D_out_channels=32,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv2  = layers.conv(C_in_channels=32,    D_out_channels=32,   K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=32, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv2 = layers.conv(C_in_channels=32,    D_out_channels=16,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')

        self.epw_conv3 = layers.conv(C_in_channels=16,    D_out_channels=96,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv3  = layers.conv(C_in_channels=96,    D_out_channels=96,   K_kernel_dimension=3, stride=2, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=96, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv3 = layers.conv(C_in_channels=96,    D_out_channels=24,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv4 = layers.conv(C_in_channels=24,    D_out_channels=144,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv4  = layers.conv(C_in_channels=144,   D_out_channels=144,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=144, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv4 = layers.conv(C_in_channels=144,   D_out_channels=24,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')

        self.epw_conv5 = layers.conv(C_in_channels=24,    D_out_channels=144,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv5  = layers.conv(C_in_channels=144,   D_out_channels=144,  K_kernel_dimension=3, stride=2, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=144, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv5 = layers.conv(C_in_channels=144,   D_out_channels=32,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv6 = layers.conv(C_in_channels=32,    D_out_channels=192,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv6  = layers.conv(C_in_channels=192,   D_out_channels=192,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=192, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv6 = layers.conv(C_in_channels=192,   D_out_channels=32,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv7 = layers.conv(C_in_channels=32,    D_out_channels=192,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv7  = layers.conv(C_in_channels=192,   D_out_channels=192,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=192, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv7 = layers.conv(C_in_channels=192,   D_out_channels=32,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')

        self.epw_conv8  = layers.conv(C_in_channels=32,   D_out_channels=192,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv8   = layers.conv(C_in_channels=192,  D_out_channels=192,  K_kernel_dimension=3, stride=2, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=192, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv8  = layers.conv(C_in_channels=192,  D_out_channels=64,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv9  = layers.conv(C_in_channels=64,   D_out_channels=384,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv9   = layers.conv(C_in_channels=384,  D_out_channels=384,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=384, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv9  = layers.conv(C_in_channels=384,  D_out_channels=64,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv10 = layers.conv(C_in_channels=64,   D_out_channels=384,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv10  = layers.conv(C_in_channels=384,  D_out_channels=384,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=384, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv10 = layers.conv(C_in_channels=384,  D_out_channels=64,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv11 = layers.conv(C_in_channels=64,   D_out_channels=384,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv11  = layers.conv(C_in_channels=384,  D_out_channels=384,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=384, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv11 = layers.conv(C_in_channels=384,  D_out_channels=64,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        
        self.epw_conv12  = layers.conv(C_in_channels=64,  D_out_channels=384,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv12   = layers.conv(C_in_channels=384, D_out_channels=384,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=384, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv12  = layers.conv(C_in_channels=384, D_out_channels=96,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv13  = layers.conv(C_in_channels=96,  D_out_channels=576,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv13   = layers.conv(C_in_channels=576, D_out_channels=576,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=576, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv13  = layers.conv(C_in_channels=576, D_out_channels=96,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv14  = layers.conv(C_in_channels=96,  D_out_channels=576,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv14   = layers.conv(C_in_channels=576, D_out_channels=576,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=576, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv14  = layers.conv(C_in_channels=576, D_out_channels=96,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained') #ilk çıkış: torch.Size([2, 96, /16, /16])

        self.epw_conv15  = layers.conv(C_in_channels=96,  D_out_channels=576,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv15   = layers.conv(C_in_channels=576, D_out_channels=576,  K_kernel_dimension=3, stride=2, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=576, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv15  = layers.conv(C_in_channels=576, D_out_channels=160,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv16  = layers.conv(C_in_channels=160, D_out_channels=960,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv16   = layers.conv(C_in_channels=960, D_out_channels=960,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=960, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv16  = layers.conv(C_in_channels=960, D_out_channels=160,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')
        self.epw_conv17  = layers.conv(C_in_channels=160, D_out_channels=960,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv17   = layers.conv(C_in_channels=960, D_out_channels=960,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=960, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv17  = layers.conv(C_in_channels=960, D_out_channels=160,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained')

        self.epw_conv18  = layers.conv(C_in_channels=160, D_out_channels=960,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', quantization_mode = 'fpt_unconstrained')
        self.dw_conv18   = layers.conv(C_in_channels=960, D_out_channels=960,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, bias=False, activation='relu', num_groups=960, quantization_mode = 'fpt_unconstrained')
        self.ppw_conv18  = layers.conv(C_in_channels=960, D_out_channels=320,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), batchnorm=True, batchnorm_affine=True, bias=False, quantization_mode = 'fpt_unconstrained') #ikinci çıkış: torch.Size([2, 320, /32, /32])

        self.head1_dw_classification = layers.conv(C_in_channels=96,  D_out_channels=96,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, activation='relu', num_groups=96, quantization_mode = 'fpt_unconstrained')
        self.head1_pw_classification = layers.conv(C_in_channels=96,  D_out_channels=8,   K_kernel_dimension=1, stride=1, padding=(0,0,0,0), output_width_30b = True, quantization_mode = 'fpt_unconstrained')
        self.head1_dw_regression     = layers.conv(C_in_channels=96,  D_out_channels=96,  K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, activation='relu', num_groups=96, quantization_mode = 'fpt_unconstrained')
        self.head1_pw_regression     = layers.conv(C_in_channels=96,  D_out_channels=16,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), output_width_30b = True, quantization_mode = 'fpt_unconstrained')

        self.head2_dw_classification = layers.conv(C_in_channels=320, D_out_channels=320, K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, activation='relu', num_groups=320, quantization_mode = 'fpt_unconstrained')
        self.head2_pw_classification = layers.conv(C_in_channels=320, D_out_channels=10,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), output_width_30b = True, quantization_mode = 'fpt_unconstrained')
        self.head2_dw_regression     = layers.conv(C_in_channels=320, D_out_channels=320, K_kernel_dimension=3, stride=1, padding=(1,1,1,1), batchnorm=True, batchnorm_affine=True, activation='relu', num_groups=320, quantization_mode = 'fpt_unconstrained')
        self.head2_pw_regression     = layers.conv(C_in_channels=320, D_out_channels=20,  K_kernel_dimension=1, stride=1, padding=(0,0,0,0), output_width_30b = True, quantization_mode = 'fpt_unconstrained')
        
        self.add_residual            = layers.add_residual(quantization_mode = 'fpt_unconstrained')


    def forward(self, x):
        x = self.conv1(x)

        x = self.epw_conv2(x)
        x = self.dw_conv2(x)
        x = self.ppw_conv2(x)

        x = self.epw_conv3(x)
        x = self.dw_conv3(x)
        x = self.ppw_conv3(x)
        res4 = x
        x = self.epw_conv4(x)
        x = self.dw_conv4(x)
        x = self.ppw_conv4(x)
        x = self.add_residual(x,res4)

        x = self.epw_conv5(x)
        x = self.dw_conv5(x)
        x = self.ppw_conv5(x)
        res6 = x
        x = self.epw_conv6(x)
        x = self.dw_conv6(x)
        x = self.ppw_conv6(x)
        x = self.add_residual(x,res6)
        res7 = x
        x = self.epw_conv7(x)
        x = self.dw_conv7(x)
        x = self.ppw_conv7(x)
        x = self.add_residual(x,res7)

        x = self.epw_conv8(x)
        x = self.dw_conv8(x)
        x = self.ppw_conv8(x)
        res9 = x
        x = self.epw_conv9(x)
        x = self.dw_conv9(x)
        x = self.ppw_conv9(x)
        x = self.add_residual(x,res9)
        res10 = x
        x = self.epw_conv10(x)
        x = self.dw_conv10(x)
        x = self.ppw_conv10(x)
        x = self.add_residual(x,res10)
        res11 = x
        x = self.epw_conv11(x)
        x = self.dw_conv11(x)
        x = self.ppw_conv11(x)
        x = self.add_residual(x,res11)

        x = self.epw_conv12(x)
        x = self.dw_conv12(x)
        x = self.ppw_conv12(x)
        res13 = x
        x = self.epw_conv13(x)
        x = self.dw_conv13(x)
        x = self.ppw_conv13(x)
        x = self.add_residual(x,res13)
        res14 = x
        x = self.epw_conv14(x)
        x = self.dw_conv14(x)
        x = self.ppw_conv14(x)
        x = self.add_residual(x,res14)
        output1 = x

        x = self.epw_conv15(x)
        x = self.dw_conv15(x)
        x = self.ppw_conv15(x)
        res16 = x
        x = self.epw_conv16(x)
        x = self.dw_conv16(x)
        x = self.ppw_conv16(x)
        x = self.add_residual(x,res16)
        res17 = x
        x = self.epw_conv17(x)
        x = self.dw_conv17(x)
        x = self.ppw_conv17(x)
        x = self.add_residual(x,res17)

        x = self.epw_conv18(x)
        x = self.dw_conv18(x)
        x = self.ppw_conv18(x)
        output2 = x

        output1_class = self.head1_dw_classification(output1)
        output1_class = self.head1_pw_classification(output1_class)
        output1_reg   = self.head1_dw_regression(output1)
        output1_reg   = self.head1_pw_regression(output1_reg)

        output2_class = self.head2_dw_classification(output2)
        output2_class = self.head2_pw_classification(output2_class)
        output2_reg   = self.head2_dw_regression(output2)
        output2_reg   = self.head2_pw_regression(output2_reg)

        #print(f"Output1 Regression: {output1_reg.shape}, Output1 Classification: {output1_class.shape}\nOutput2 Regression: {output2_reg.shape}, Output2 Classification: {output2_class.shape}")
        return (output1_reg, output1_class, output2_reg, output2_class)