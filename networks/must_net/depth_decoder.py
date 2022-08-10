from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from networks.layers import *

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, numT=None):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.numT = numT

        if numT == None:
            self.weight_decoder = weight_decoder(self.num_ch_enc, num_input_features=1, num_layers_to_predict=32,
                                                 stride=1)
        else:
            self.weight_decoder = nn.ModuleDict()
            for T_idx in range(numT):
                self.weight_decoder["weight_{}".format(T_idx)] = weight_decoder(self.num_ch_enc, num_input_features=1,
                                                                 num_layers_to_predict=32,
                                                                 stride=1)
        self.convs = nn.ModuleDict()

        for j in range(5):
            for i in range(5 - j):
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

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

    def forward(self, input_features,frame_index):
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

        if self.numT == None:
            pred_weights = self.weight_decoder(input_features)
            for layer in range(4):
                outputs[("disp",frame_index, layer,0)] = torch.sum(((outputs[("disp_s", layer)]) * pred_weights), dim=1,
                                                              keepdim=True)
            outputs[("weights")] = pred_weights
            outputs[("norm_constraint")] = 0
        else:
            pred_weights = {}
            outputs[("norm_constraint")] = 0
            for T_idx in range(self.numT):
                pred_weights["T_{}".format(T_idx)] = self.weight_decoder["weight_{}".format(T_idx)](input_features)
                for layer in range(4):
                    outputs[("disp", frame_index, layer, T_idx)] = torch.sum(
                        ((outputs[("disp_s", layer)]) * pred_weights["T_{}".format(T_idx)]), dim=1,
                        keepdim=True)
                outputs[("weights_{}".format(T_idx))] = pred_weights["T_{}".format(T_idx)]
            cat_disp = []
            for T_idx in range(self.numT):
                cat_disp.append(outputs[("disp", frame_index, 0, T_idx)])
            cat_disp = torch.cat(cat_disp,dim=1)

            # for bases diversity constraints
            mean = torch.mean(torch.mean(cat_disp,dim=-2,keepdim=True),dim=-1,keepdim=True)
            var = torch.mean(
                torch.mean(torch.abs(cat_disp - mean) ** 2, dim=-1, keepdim=True), dim=-2,
                keepdim=True)
            norm_constraint = 1 / (torch.mean(
                                torch.mean(mean * mean, dim=1, keepdim=True) - (
                                    torch.mean(mean, dim=1, keepdim=True)) ** 2) + 1e-3)
            norm_constraint *= (torch.mean(
                torch.mean(var * var, dim=1, keepdim=True) - (
                    torch.mean(var, dim=1, keepdim=True)) ** 2) + 1e-3)

            outputs[("norm_constraint")] += norm_constraint

        return outputs

class weight_decoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_layers_to_predict=None, stride=1):
        super(weight_decoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.num_layers_to_predict_for = num_layers_to_predict

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("depth", 0)] = nn.Conv2d(num_input_features*2 * 256, 256, 3, stride, 1)
        self.convs[("depth", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("depth", 2)] = nn.Conv2d(256, num_layers_to_predict, 1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

        self.init_weights()

    def forward(self, input_features):
        """
        Return the weights of a single stream
        """
        out = input_features[-1]
        for i in range(3):
            out = self.convs[("depth", i)](out)
            if i!=2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        weights = out.view(-1, self.num_layers_to_predict_for, 1, 1)
        return weights
    def init_weights(self):
        """
        Initializes network weights.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
