# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.models.layers import tensor_fusion_helper as helper


class UAFM(nn.Layer):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        # print("Now prepare x:")
        x = self.prepare_x(x, y)
       #  print("Now prepare y:")
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_ChAtten_S(UAFM):
    """
    The UAFM with channel attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                2 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        #print("Now use Spa atte")
        #print(x.shape,y.shape)
        atten = helper.avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))
        
        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten_S(UAFM):
    """
    The UAFM with spatial attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(
                2, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out



class UAFM_Union_Atten(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        # low
        self.x_ch = x_ch
        # high
        self.y_ch = y_ch
        self.out_ch = out_ch
        self.sec_out_ch = 128
        self.sec_out_thr = 64
        self.conv_out_sec = layers.ConvBNReLU(y_ch, self.sec_out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.conv_out_thr = layers.ConvBNReLU(y_ch, self.sec_out_thr, kernel_size=3, padding=1, bias_attr=False)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))
        
        self.conv_xy_atten_SPA = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))
    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        if self.out_ch == 64:
            out = self.conv_out_sec(out)
        elif self.out_ch == 32:
            out = self.conv_out_thr(out)
        else:
            out = self.conv_out(out)
        return out
    
    
    def Spafuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        #print(x.shape,y.shape)
        #print("THe xch/ych/ourch is",self.x_ch,",",self.y_ch,",",self.out_ch)
        #print("Now start avg_max")
        atten = helper.avg_max_reduce_channel([x, y])
        #print("Now start convy")
        atten = F.sigmoid(self.conv_xy_atten_SPA(atten))
        #print("Now start add")
        out = x * atten + y * (1 - atten)
        #print("Now end add")
        out = self.conv_out(out)
        return out
    
    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        orgx = x
        #print("Now the x is:",x.shape,"The y is ",y.shape)
        x, y = self.prepare(x, y)
        #print("After prepare x is",x.shape,"y is",y.shape)
        out = self.fuse(x, y)
        #print("The out's shape is:",out.shape,"The y's shape is :",y.shape)

        x,y = self.prepare(orgx,out)
        #print("After sec prepare x is",x.shape,"y is",y.shape)
        #print("Now start Spfuse")
        UnionOut = self.Spafuse(x,y)
        #print("Now the out shape is",UnionOut.shape)
        return UnionOut

class UAFM_Union_Atten_S(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        # low
        self.x_ch = x_ch
        # high
        self.y_ch = y_ch
        self.out_ch = out_ch
        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        # 1024 128 128
        # ppliteseg 对注意力机制的优化，在于每一次输出的形状是相等的
        print("Now this is the Cha's parm:",self.x_ch,self.y_ch,self.out_ch)
        print("The X shape:",x.shape)
        print("The y shape:",y.shape)
        atten = helper.avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))
    
        out = x * atten + y * (1 - atten)
        print("Now the un conv's out shape is:",out.shape)
        out = self.conv_out(out)
        print("Now the output' shape is :",out.shape)
        print("Now prepare the Spatten:")
        Spa_Att = UAFM_SpAtten(self.x_ch, self.y_ch, self.out_ch ,ksize=3)
        union_out = Spa_Att(x,out)
        return union_out
