import torch
import torch.nn as nn
from . import networks


# もしかしてnn.Moduleのサブクラスでなくてもいい？
class UNIT(nn.Module):
    def __init__(self, input_dim, dim, num_down, norm, activation, num_resblock, output_dim, pad_type):

        super(UNIT, self).__init__()
        # self.vae_a = vaeGenerator()
        # self.
        # 設定ファイル関連どうする？
        """
        norm:normalization layer
        """
        self.enc_a = UnitEncoder(input_dim, dim, num_down, 3, pad_type="zero", norm=norm, activation="lrelu")
        self.enc_b = UnitEncoder(input_dim, dim, num_down, 3, pad_type="zero", norm=norm, activation="lrelu")
        self.share_enc = ResBlock(
            self.enc_a.output_dim, norm=norm, activation="lrelu", pad_type=pad_type)
        self.share_dec = ResBlock(
            self.enc_a.output_dim, norm=norm, activation="lrelu", pad_type=pad_type)
        self.dec_a = UnitDecoder(output_dim, self.enc_a.output_dim, num_down, 3, pad_type="zero", norm=norm, activation="lrelu")
        self.dec_b = UnitDecoder(output_dim, self.enc_a.output_dim, num_down, 3, pad_type="zero", norm=norm, activation="lrelu")

    def forward(self, x):
        # Should I implement?
        pass

    def encode_a(self, x):
        hidden = self.share_enc(self.enc_a(x))
        noise = torch.rand_like(hidden)
        return hidden, noise

    def encode_b(self, x):
        hidden = self.share_enc(self.enc_b(x))
        noise = torch.rand_like(hidden)
        return hidden, noise

    def decode_a(self, z):
        images = self.dec_a(self.share_dec(z))
        return images

    def decode_b(self, z):
        images = self.dec_b(self.share_dec(z))
        return images


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()

    def forward(self, x):
        pass

#########################
# Encoder and Decoders
#########################


class UnitEncoder(nn.Module):
    def __init__(self, input_dim, dim, num_down, num_resblock, pad_type="zero", norm="bn", activation="lrelu"):
        super(UnitEncoder, self).__init__()
        model = []
        model += [Conv2dBlock(input_dim, dim,
                              kernel_size=4, stride=2, padding=1, pad_type=pad_type, norm=norm, activation=activation)]
        # self.tmp = nn.Sequential(*model)
        for i in range(num_down):
            model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                  pad_type=pad_type, norm=norm, activation=activation)]
            dim *= 2
        self.output_dim = dim

        model += [ResBlocks(num_resblock, dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        # tmp = self.tmp(x)
        # print("before res", tmp.size())
        return self.model(x)


class UnitDecoder(nn.Module):
    def __init__(self, output_dim, dim, num_up, num_resblock, pad_type="zero", norm="bn", activation="lrelu"):
        super(UnitDecoder, self).__init__()
        model = []
        # weight shareすべき
        model += [ResBlocks(num_resblock, dim, norm=norm, activation=activation, pad_type=pad_type)]

        for i in range(num_up):
            model += [nn.Upsample(scale_factor=2), Conv2dBlock(dim, dim // 2,
                                  5, 1, 2, pad_type=pad_type, norm=norm, activation=activation)]
            dim //= 2
        model += [nn.Upsample(scale_factor=2), Conv2dBlock(dim, output_dim, 7, 1, 3,
                              pad_type=pad_type, norm=norm, activation="tanh")]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


#########################
# Sequential Models
#########################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm="instance", activation="lrelu", pad_type="zero"):
        super(ResBlocks, self).__init__()
        model = []
        for i in range(num_blocks):
            model += [ResBlock(dim, norm=norm,
                               activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#########################
# Basic Blocks
#########################


class ResBlock(nn.Module):
    def __init__(self, dim, norm="instance", activation="lrelu", pad_type="zero"):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, pad_type=pad_type,
                              norm=norm, activation=activation)]

        model += [Conv2dBlock(dim, dim, 3, 1, 1, pad_type=pad_type,
                              norm="none", activation=activation)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x) + x
        return output


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, pad_type="zero", norm="bn", activation="lrelu"):
        super(Conv2dBlock, self).__init__()
        conv_block = []
        self.use_bias = True
        if pad_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(padding)]
        elif pad_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(padding)]
        elif pad_type == 'zero':
            conv_block += [nn.ZeroPad2d(padding)]
        else:
            raise NotImplementedError(
                'Padding [%s] is not implemented.' % pad_type)
        conv_block += [nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)]

        if norm == "bn":
            conv_block += [nn.BatchNorm2d(output_dim)]
        elif norm == "instance":
            conv_block += [nn.InstanceNorm2d(output_dim)]
        elif norm == "sepctral":
            "Implement later"
            pass
        elif norm == "none":
            conv_block = conv_block
        else:
            raise NotImplementedError(
                "Normalization layer [%s] is not implemented." % norm)

        if activation == "lrelu":
            conv_block += [nn.LeakyReLU(0.2, inplace=True)]
        elif activation == "tanh":
            conv_block += [nn.Tanh()]
        elif activation == "none":
            conv_block = conv_block
        else:
            raise NotImplementedError(
        "Activation [%s] is not implemented." % activation)

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.model(x)



# 以下必要なし
from torch.nn import functional as F


class naive_VAE(nn.Module):
    def __init__(self):
        super(naive_VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # -1なんで必要
        # mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x.view(-1, 784))
        self.z = self.reparameterize(mu, logvar)
        return self.decode(self.z), mu, logvar

    def save(self, which_epoch):
        self.save_network(self.unit, "unit", which_epoch, self.gpu_ids)
        self.save_network(self.net_D, "", which_epoch, self.gpu_ids)