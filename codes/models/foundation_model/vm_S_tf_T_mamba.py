import os.path
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from mamba_ssm import Mamba
import torch.nn.init as init
from einops.layers.torch import Rearrange

VECTOR_LEN = 512


class CNNBlocker(nn.Module):
    def __init__(self, in_c):
        super(CNNBlocker, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.nb = nn.BatchNorm2d(in_c)
        self.activate = nn.Tanh()

    def forward(self, x):
        residual_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual_x
        x = self.nb(x)
        x = self.activate(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class MulHeadAttentionBlocker(nn.Module):
    def __init__(self, in_c, patch_h, patch_w, n_heads=8):
        super(MulHeadAttentionBlocker, self).__init__()
        self.ph = patch_h
        self.pw = patch_w
        embed_dim = patch_h * patch_w
        self.mul_head_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)
        # self.norm1 = nn.LayerNorm(embed_dim)
        self.nb = nn.BatchNorm2d(in_c)
        self.activate = nn.Tanh()

    def forward(self, x):
        b, c, h, w = x.size()
        # print("x-1:", x.size())
        residual_x = x
        nh = h // self.ph
        nw = w // self.pw
        rearrange = Rearrange('b c (nh ph) (nw pw) -> (c nh nw) b (ph pw)', c=c, ph=self.ph, pw=self.pw, nh=nh, nw=nw)
        # print("x0:", x.size())
        x = rearrange(x)
        # print("x1:", x.size())
        x = self.mul_head_attn(x, x, x)[0]
        # x = self.norm1(x)
        # print("x2:", x.size())
        un_rearrange = Rearrange('(c nh nw) b (ph pw) -> b c (nh ph) (nw pw)', c=c, ph=self.ph, pw=self.pw, nh=nh,
                                 nw=nw)
        x = un_rearrange(x)
        # x = x + x1
        # print("x3:", x.size())

        # print("x4:", x.size())
        # sys.exit()

        x = x + residual_x
        x = self.nb(x)
        x = self.activate(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class SpaceBlocker(nn.Module):
    def __init__(self, in_c):
        super(SpaceBlocker, self).__init__()

        self.mha_b1 = MulHeadAttentionBlocker(in_c, 8, 8, 8)
        # self.cnn_b1 = CNNBlocker(in_c)
        self.conv1 = nn.Conv2d(in_c * 2, in_c * 2, kernel_size=(4, 4), stride=(2, 2), padding=1)

        self.mha_b2 = MulHeadAttentionBlocker(in_c * 2, 8, 8, 8)
        # self.cnn_b2 = CNNBlocker(in_c * 2)
        self.conv2 = nn.Conv2d(in_c * 4, in_c * 4, kernel_size=(4, 4), stride=(2, 2), padding=1)

        self.mha_b3 = MulHeadAttentionBlocker(in_c * 4, 4, 4, 16)
        # self.cnn_b3 = CNNBlocker(in_c * 4)
        self.conv3 = nn.Conv2d(in_c * 8, in_c * 8, kernel_size=(4, 4), stride=(2, 2), padding=1)

        self.mha_b4 = MulHeadAttentionBlocker(in_c * 8, 4, 4, 16)
        # self.cnn_b4 = CNNBlocker(in_c * 8)
        self.conv4 = nn.Conv2d(in_c * 16, in_c * 16, kernel_size=(4, 4), stride=(2, 2), padding=1)
        # self.mam_ba5 = MulHeadAttentionBlocker(in_c * 16, 64)
        # self.cnn_ba5 = CNNBlocker(in_c * 16)

    def forward(self, x):
        # b, c, h, w = x.size()
        # mha1 = MulHeadAttentionBlocker(c, h // 4 * w // 4 * c, b*10, 16)
        x_m = self.mha_b1(x)
        # x_c = self.cnn_b1(x)
        x = torch.cat((x_m, x_m), dim=1)
        x = self.conv1(x)

        x_m = self.mha_b2(x)
        # x_c = self.cnn_b2(x)
        x = torch.cat((x_m, x_m), dim=1)
        x = self.conv2(x)

        x_m = self.mha_b3(x)
        # x_c = self.cnn_b3(x)
        x = torch.cat((x_m, x_m), dim=1)
        x = self.conv3(x)

        x_m = self.mha_b4(x)
        # x_c = self.cnn_b4(x)
        x = torch.cat((x_m, x_m), dim=1)
        x = self.conv4(x)
        # print(x.size())
        # sys.exit()
        # x_m = self.mam_ba5(x)
        # x_c = self.cnn_ba5(x)
        return x


class TimeBlocker(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeBlocker, self).__init__()
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=48*8*8,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size, 1024)
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.mamba(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activate(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.space = SpaceBlocker(3)
        self.timer = TimeBlocker(48 * 8 * 8, VECTOR_LEN * 2)  # 512

    def forward(self, x):
        b, c, l, h, w = x.size()
        x = torch.transpose(x, 1, 2)
        x = x.reshape(b * l, c, h, w)
        x = self.space(x)  # torch.Size([40, 48, 8, 8])
        # print("self.space(x):", x.size())
        x = x.reshape(b, l, 48 * 8 * 8)  # torch.Size([4, 10, 3072])
        # print("x = x.reshape(b, l, 48 * 8 * 8):", x.size())
        # x = torch.transpose(x, 1, 2)  # torch.Size([4, 3072, 10])
        # print("torch.transpose(x, 1, 2):", x.size())
        # sys.exit()
        x = self.timer(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class TimeExtern(nn.Module):
    def __init__(self):
        super(TimeExtern, self).__init__()
        self.conv2d_1 = nn.Conv2d(20, 20, 3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(20, 10, 3, stride=1, padding=1)
        # self.conv2d_3 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        # self.conv2d_4 = nn.Conv2d(8, 10, 3, stride=1, padding=1)
        # self.liner1 = nn.Linear(VECTOR_LEN, VECTOR_LEN*10)
        self.activate = nn.Tanh()

    def forward(self, x):
        # print(x.size())
        # x = self.fc(x)
        b, c, l = x.size()
        x = x.reshape(b, c, 32, 32)
        # x = x.repeat(1, 10, 1)
        # print(x.size())
        # sys.exit()
        # lstm_out, (_, _) = self.lstm(x)
        # x = self.liner1(x)
        # print(x.shape)
        # sys.exit()

        # x = torch.transpose(x, 0, 1)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        # x = self.conv2d_3(x)
        # x = self.conv2d_4(x)
        # print(x.size())
        # x = x.reshape(b, 10, 16, 16)
        # x = self.activate(x)
        # print(x.size())
        # sys.exit()
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.fc1 = nn.Linear(VECTOR_LEN * 2, 64 * 10 * 16 * 16)
        # self.fc1 = nn.Linear(VECTOR_LEN * 2, VECTOR_LEN * 20)
        self.time2D = TimeExtern()
        self.deconv0 = nn.ConvTranspose3d(1, 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.relu1 = nn.ReLU()
        self.deconv1 = nn.ConvTranspose3d(2, 6, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.relu2 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose3d(6, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        # self.deconv3 = nn.ConvTranspose3d(6, 3, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.relu2 = nn.ReLU()
        self.activate = nn.Tanh()
        # self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.time2D(x)
        # print('self.time2D(x):', x.size())
        # sys.exit()
        x = x.view(x.size(0), 1, 10, 32, 32)  # Reshape to feature map
        # print(x.size())
        # sys.exit()
        x = self.deconv0(x)
        # x = self.relu1(x)
        x = self.deconv1(x)
        # x = self.relu2(x)
        x = self.deconv2(x)

        # x = self.deconv3(x)
        x = self.activate(x)  # 使用sigmoid以确保输出在[0, 1]范围内
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class ShareInputBlocker(nn.Module):
    def __init__(self):
        super(ShareInputBlocker, self).__init__()
        self.fc1 = nn.Linear(10, 1)  # 1024*10->1024
        self.dp1 = nn.Dropout(0.2)
        # self.act_fun = nn.ReLU()
        # self.fc2 = nn.Linear(VECTOR_LEN * 2, VECTOR_LEN)
        # self.dp2 = nn.Dropout(0.2)
        self.activate = nn.Tanh()

    def forward(self, x):
        print(x.size())  # torch.Size([4, 10, 1024])
        x = torch.transpose(x, -1, -2)
        print(x.size())  # torch.Size([4, 1024, 10])
        x = self.dp1(self.fc1(x))
        print(x.size())  # torch.Size([4, 1024, 1])
        x = torch.transpose(x, -1, -2)
        print("aaa:", x.size())  # torch.Size([4, 1, 1024])
        # sys.exit()
        x = self.activate(x)
        # x = self.activate(self.dp2(self.fc2(x)))
        return x


class ShareOutputBlocker(nn.Module):
    def __init__(self):
        super(ShareOutputBlocker, self).__init__()
        # self.fc0 = nn.Linear(10, 1)
        self.fc1 = nn.Linear(VECTOR_LEN * 2, VECTOR_LEN * 4)
        self.dp1 = nn.Dropout(0.2)
        # self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(VECTOR_LEN * 4, VECTOR_LEN * 2)
        self.dp2 = nn.Dropout(0.2)
        # self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(VECTOR_LEN * 2, VECTOR_LEN * 2)
        self.dp3 = nn.Dropout(0.2)
        self.activate = nn.Tanh()

    def forward(self, x):
        # x = torch.transpose(x, -1, -2)
        # x = self.fc0(x)
        # x = torch.transpose(x, -1, -2)
        x = self.dp1(self.fc1(x))
        x = self.dp2(self.fc2(x))
        x = self.dp2(self.fc3(x))
        x = self.activate(x)
        return x


class ShareExpert(nn.Module):
    def __init__(self):
        super(ShareExpert, self).__init__()
        # self.input_blocker_list = nn.ModuleList([ShareInputBlocker() for _ in range(4)])
        self.output_blocker = ShareOutputBlocker()

    def forward(self, x, mask_code):
        # for i in range(4):
        #     if mask_code[i] == 1:
        #         x = self.input_blocker_list[i](x)
        x = self.output_blocker(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class IndependentExpert(nn.Module):
    def __init__(self):
        super(IndependentExpert, self).__init__()
        # self.fc0 = nn.Linear(10, 1)

        self.fc1 = nn.Linear(VECTOR_LEN * 2, VECTOR_LEN * 4)
        self.dp1 = nn.Dropout(0.3)
        # self.act_fun = nn.ReLU()
        self.fc2 = nn.Linear(VECTOR_LEN * 4, VECTOR_LEN * 2)
        self.dp2 = nn.Dropout(0.3)
        self.activate = nn.Tanh()
        # self.fc3 = nn.Linear(VECTOR_LEN * 2, VECTOR_LEN)
        # self.dp3 = nn.Dropout(0.3)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # x = torch.transpose(x, -1, -2)
        # x = self.fc0(x)
        # x = torch.transpose(x, -1, -2)
        x = self.dp1(self.fc1(x))
        x = self.dp2(self.fc2(x))
        # print("IndependentExpert:", x.size())
        x = self.activate(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class MulModalityMaeMoe(nn.Module):
    def __init__(self):
        super(MulModalityMaeMoe, self).__init__()
        self.encoder_list = nn.ModuleList([Encoder() for _ in range(4)])  # us, ue, ce , cf
        self.decoder_list = nn.ModuleList([Decoder() for _ in range(4)])  # us, ue, ce , cf
        self.share_expert = ShareExpert()
        self.independentExpert_list = nn.ModuleList([IndependentExpert() for _ in range(4)])

    def forward(self, x, mask_code):
        b, c, l, h, w = x.size()
        # print(b, c, l, h, w)
        # sys.exit()
        # print('mask_code:', mask_code)
        # inputs_list = x[]
        encoded_i_list = torch.zeros(b, 40, VECTOR_LEN * 2)
        encoded_s_list = torch.zeros(b, 40, VECTOR_LEN * 2)
        encoded_list = torch.zeros(b, 40, VECTOR_LEN * 2)

        for i in range(4):
            if mask_code[i] == 1:
                inputs_x = x[:, :, i * 10:(i + 1) * 10, :, :]
                # print("inputs_x:", inputs_x.size())
                encoded = self.encoder_list[i](inputs_x)  # torch.Size([4, 10, 512])
                encoded_list[:, i * 10:(i + 1) * 10, :] = encoded
                # print("encoded:", encoded.size())
                # sys.exit()
                encoded_s = self.share_expert(encoded, mask_code)
                # print("encoded_s:", encoded_s.size())
                # sys.exit()
                encoded_i = self.independentExpert_list[i](encoded)
                # print('encoded_i:', encoded_i.size())

                encoded_i_list[:, i * 10:(i + 1) * 10, :] = encoded_i
                encoded_s_list[:, i * 10:(i + 1) * 10, :] = encoded_s
                encoded_i_s = torch.cat((encoded_i, encoded_s), dim=1)
                # print(encoded_i_s.size())
                # sys.exit()
                # encoded_all = encoded_i * encoded + encoded_s
                # encoded_all = torch.cat((encoded, encoded_i_s), dim=1)
                # print(encoded_all.size())
                # sys.exit()
                x[:, :, i * 10:(i + 1) * 10, :, :] = self.decoder_list[i](encoded_i_s)
                # encoded_i_list = encoded_i_list.reshape(b, 4, VECTOR_LEN * 2 * 10)
                # encoded_s_list = encoded_s_list.reshape(b, 4, VECTOR_LEN * 2 * 10)
        return x, encoded_s_list, encoded_i_list, encoded_list

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def save_encoder_models(self, file_path, file_head):
        for i in range(4):
            encoder_file_name = os.path.join(file_path, file_head + ("_encoder_%d.pth" % i))
            torch.save(self.encoder_list[i].state_dict(), encoder_file_name)
        print("save_encoder_models_ok")

    def save_decoder_models(self, file_path, file_head):
        for i in range(4):
            decoder_file_name = os.path.join(file_path, file_head + ("_decoder_%d.pth" % i))
            torch.save(self.decoder_list[i].state_dict(), decoder_file_name)
        print("save_decoder_models_ok")

    def save_models(self, file_path, file_head):
        model_dir = os.path.join(file_path, file_head + ".pth")
        print("save_model:", model_dir)
        torch.save(self.state_dict(), model_dir)
        # self.save_encoder_models(file_path, file_head)
        # self.save_decoder_models(file_path, file_head)

    #
    def load_encoder_models(self, file_path, file_head):
        for i in range(4):
            encoder_file_name = os.path.join(file_path, file_head + ("_encoder_%d.pth" % i))
            self.encoder_list[i].load_state_dict(torch.load(encoder_file_name))
        print("load_encoder_models_ok")

    def load_decoder_models(self, file_path, file_head):
        for i in range(4):
            decoder_file_name = os.path.join(file_path, file_head + ("_decoder_%d.pth" % i))
            self.encoder_list[i].load_state_dict(torch.load(decoder_file_name))
        print("load_decoder_models_ok")

    def load_models(self, file_path, file_head):
        model_dir = os.path.join(file_path, file_head + ".pth")
        print("load_model:", model_dir)
        self.load_state_dict(torch.load(model_dir))
