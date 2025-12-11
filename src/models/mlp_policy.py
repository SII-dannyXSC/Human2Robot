import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 Two-Layer MLP 模型
class OneLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        # 第一层：线性层 followed by ReLU
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


# 定义 Two-Layer MLP 模型
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()

        # BatchNorm for input
        self.batch_norm = nn.BatchNorm1d(input_dim)

        # 第一层：线性层 followed by ReLU
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 第二层：线性层
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过 BatchNorm 进行输入标准化
        x = self.batch_norm(x)

        # 第一层：线性层 + ReLU
        x = F.relu(self.fc1(x))

        # 第二层：线性层（输出）
        x = self.fc2(x)

        return x


# class WHMLP(nn.Module):
#     def __init__(
#         self,
#     ):
#         super(WHMLP, self).__init__()

#         conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)


class ZHMLP(nn.Module):
    def __init__(
        self,
        channel_dim,
        wh_dim,
        hidden_dim,
        output_dim,
    ):
        super(ZHMLP, self).__init__()
        self.encoder_qpos_proj = nn.Linear(output_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(channel_dim)
        self.fc1 = nn.Linear(channel_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc3 = nn.Linear(wh_dim + 1, 1)

    def forward(self, hf, qpos):
        # qpos: end position of the first frame b 7
        qpos = torch.unsqueeze(qpos, axis=-2)  # b 1 7

        if len(hf.size()) == 5:
            f = hf.size()[2]
            qpos = qpos.repeat(1, f, 1)  # b f 7
        elif len(hf.size()) == 4:
            f = hf.size()[1]
            qpos = qpos.repeat(f, 1)  # f 7

        qpos_embed = self.encoder_qpos_proj(qpos)  # b f hid_dim
        qpos_embed = torch.unsqueeze(qpos_embed, axis=-2)  # b f 1 hid_dim

        # hf: b c f h w
        hf = hf.transpose(-3, -4)  # b f c h w
        hf = hf.flatten(-2).transpose(-1, -2)  # b f h*w c

        hf = self.norm1(hf)
        hf = F.silu(hf)
        hf = self.fc1(hf)  # b f h*w hid_dim
        # print(f"hf: {hf.size()}")
        # print(f"qpos: {qpos_embed.size()}")

        # cat
        hf = torch.cat([hf, qpos_embed], axis=-2)  # b f h*w+1 hid_dim

        hf = self.norm2(hf)
        hf = F.silu(hf)
        hf = self.fc2(hf)  # b f h*w+1 output_dim

        hf = hf.transpose(-1, -2)  # b f output_dim h*w+1

        hf = F.silu(hf)
        hf = self.fc3(hf)  # b f output_dim 1

        hf = hf.squeeze(-1)  # b f output_dim
        return hf


class TestMLP(nn.Module):
    def __init__(
        self,
        channel_dim,
        wh_dim,
        hidden_dim,
        output_dim,
    ):
        super(TestMLP, self).__init__()
        self.encoder_qpos_proj = nn.Linear(output_dim, hidden_dim)

        self.conv2 = nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv1 = nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.norm1 = nn.LayerNorm(channel_dim)
        self.fc1 = nn.Linear(channel_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc3 = nn.Linear(wh_dim + 1, 1)

    def forward(self, hf, qpos):
        have_batch = False
        # qpos: end position of the first frame b 7
        qpos = torch.unsqueeze(qpos, axis=-2)  # b 1 7

        if len(hf.size()) == 5:
            have_batch = True
            b, c, f, h, w = hf.size()
            qpos = qpos.repeat(1, f, 1)  # b f 7
            qpos = qpos.reshape(b * f, -1)
        elif len(hf.size()) == 4:
            have_batch = False
            c, f, h, w = hf.size()
            qpos = qpos.repeat(f, 1)  # f 7

        qpos_embed = self.encoder_qpos_proj(qpos)  # b*f hid_dim
        qpos_embed = torch.unsqueeze(qpos_embed, axis=-2)  # b*f 1 hid_dim

        # hf: b*f c h w
        hf = hf.reshape(-1, c, h, w)

        # down sample
        # TODO: use conv 3d?
        hf = self.conv2(hf)
        hf = self.conv1(hf)

        hf = hf.flatten(-2).transpose(-1, -2)  # b*f h*w c

        hf = self.norm1(hf)
        hf = F.silu(hf)
        hf = self.fc1(hf)  # b*f h*w hid_dim

        # cat
        hf = torch.cat([hf, qpos_embed], axis=-2)  # b*f h*w+1 hid_dim

        hf = self.norm2(hf)
        hf = F.silu(hf)
        hf = self.fc2(hf)  # b*f h*w+1 output_dim

        hf = hf.transpose(-1, -2)  # b*f output_dim h*w+1

        hf = F.silu(hf)
        hf = self.fc3(hf)  # b*f output_dim 1

        hf = hf.squeeze(-1)  # b*f output_dim

        if have_batch:
            hf = hf.reshape(b, f, -1)

        return hf
