# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.layers import *

class fuse_Conve(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fuse_Conve, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        return self.relu(self.conv_se(features))
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, Resnet50=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.complex_backbone = Resnet50
        if Resnet50:
            self.num_ch_dec = np.array([16*4, 32*4, 64*4, 128*4, 256*4])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
        if Resnet50:
            self.weight_decoder = weight_decoder(self.num_ch_enc, num_input_features=4, num_layers_to_predict=32,
                                                 stride=1)
        else:
            self.weight_decoder = weight_decoder(self.num_ch_enc, num_input_features=1, num_layers_to_predict=32,
                                                 stride=1)
            self.weight_decoder2 = weight_decoder(self.num_ch_enc, num_input_features=1, num_layers_to_predict=32, stride=1)
            self.weight_decoder3 = weight_decoder(self.num_ch_enc, num_input_features=1, num_layers_to_predict=32,
                                                  stride=1)
            self.weight_decoder4 = weight_decoder(self.num_ch_enc, num_input_features=1, num_layers_to_predict=32,
                                                  stride=1)
        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)
                if Resnet50:
                    if i == 0 and j == 4:
                        self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in * 4, num_ch_out*4)
                        num_ch_in = num_ch_out*4
                        num_ch_out = self.num_ch_dec[i]
                        self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)
                else:
                    if i == 0 and j == 4:
                        num_ch_in = num_ch_out
                        num_ch_out = self.num_ch_dec[i]
                        self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)



        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fuse_Conve(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                              + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                                 self.num_ch_enc[row],
                                                                                 self.num_ch_dec[row + 1])
            else:
                self.convs["X_" + index + "_downsample"] = Conv1x1(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row]
                                                                   + self.num_ch_dec[row + 1] * (col - 1),
                                                                   self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2,
                                                                                 self.num_ch_dec[row + 1])
        for i in range(4):
            if Resnet50:
                if i==0:
                    self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i+1]//2, self.num_output_channels * 32)
                else:
                    self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i+1]//2, self.num_output_channels * 32)
            else:
                if i==0:
                    self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels * 32)
                else:
                    self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels * 32)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features,frame_index,templete):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](features["X_{}{}".format(row + 1, col - 1)]),
                    low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disp_s", 0)] = self.sigmoid(self.convs["dispConvScale0"](x))
        outputs[("disp_s", 1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disp_s", 2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disp_s", 3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))

        if templete is not None:
            pred_weights = self.weight_decoder(input_features)
            outputs[("disp", frame_index,0)] = torch.sum(((outputs[("disp_s", 0)])*pred_weights),dim=1,keepdim=True)+templete[("disp", 0,0)]

            outputs[("disp", frame_index,1)] = torch.sum(((outputs[("disp_s", 1)])*pred_weights),dim=1,keepdim=True)+F.interpolate(
                templete[("disp", 0,0)], [192//2, 640//2], mode="bilinear", align_corners=False)
            outputs[("disp", frame_index,2)] = torch.sum(((outputs[("disp_s", 2)])*pred_weights),dim=1,keepdim=True)+F.interpolate(
                templete[("disp", 0,0)], [192//4, 640//4], mode="bilinear", align_corners=False)
            outputs[("disp", frame_index,3)] = torch.sum(((outputs[("disp_s", 3)])*pred_weights),dim=1,keepdim=True)+F.interpolate(
                templete[("disp", 0,0)], [192//8, 640//8], mode="bilinear", align_corners=False)
        else:
            pred_weights = self.weight_decoder(input_features)
            #pred_weights2 = self.weight_decoder2(input_features)
            outputs[("disp",frame_index, 0)] = torch.sum(((outputs[("disp_s", 0)]) * pred_weights), dim=1,
                                                          keepdim=True)
            outputs[("disp",frame_index, 1)] = torch.sum(((outputs[("disp_s", 1)]) * pred_weights), dim=1,
                                                          keepdim=True)
            outputs[("disp",frame_index, 2)] = torch.sum(((outputs[("disp_s", 2)]) * pred_weights), dim=1,
                                                          keepdim=True)
            outputs[("disp",frame_index, 3)] = torch.sum(((outputs[("disp_s", 3)]) * pred_weights), dim=1,
                                                          keepdim=True)
            outputs[("weights")] = pred_weights
            #outputs[("weights2")] = pred_weights2
        return outputs

class weight_decoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_layers_to_predict=None, stride=1):
        super(weight_decoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        self.num_layers_to_predict_for = num_layers_to_predict

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("disp", 0)] = nn.Conv2d(num_input_features*2 * 256, 256, 3, stride, 1)
        self.convs[("disp", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("disp", 2)] = nn.Conv2d(256, num_layers_to_predict, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        out = input_features[-1]
        for i in range(3):
            out = self.convs[("disp", i)](out)
            if i!=2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        weights = out.view(-1, self.num_layers_to_predict_for, 1, 1)
        return weights
