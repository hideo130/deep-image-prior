import itertools
import torch
import torch.nn as nn
from . import networks
from .base_model import BaseModel


class UNITModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = []
        self.make_loss_hist_list()
        


        # if self.isTrain and self.opt.lambda_identity > 0.0
        #     visual_names_A.append('idt_A')
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """

        if self.opt.dataset_mode != 'single':
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):
        if self.opt.dataset_mode != "single":
        else:

    def optimize_parameters(self):
        pass

    def backward_D_basic(self, netD):
        pass

    def backward_G(self):



class UNIT(nn.Module):
    def __init__(self, opt):

        super(UNIT, self).__init__()
        # self.vae_a = vaeGenerator()
        # self.
        self.enc1 = MakeEncoder()
        self.enc2 = MakeEncoder()
        self.share_enc = ResnetBlock()
        self.share_dec = ResnetBlock()
        self.dec1 = make_encoder()
        self.dec2 = make_encoder()
        

    def forward(self, x):
        x = self.enc(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, pad_type="zero", norm="bn",activation="lrelu"):
        super(Conv2dBlock, self).__init__()
        self.conv_block = []
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
        
        if activation=="lrelu":
            conv_block += nn.LeakyReLU(0.2, inplace=True)
        else activation=="tanh":
            conv_block += nn.Tanh()
        else activation=="none"
            conv_block = conv_block
        else:
            raise NotImplementedError(
        "Activation [%s] is not implemented." % activation)

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.model(x)




class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()


    def forward(self, x):
        pass


class MakeEndcoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_downs, first_ch=64, norm_layer=nn.BatchNorm2d):
        self.model = []
        self.model += [Conv2dBlock(input_dim, first_ch, kernel_size=4, stride=2, padding=2 ))]
        

        for i in range(num_downs):
            down.append(i)


class ResnetBlock():
    def __init__(self, dim, norm="instance", activation="lrelu", pad_type= "zero"):
        super(ResnetBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,  pad_type=pad_type, norm=norm, activation=activation)]
        
        model += [Conv2dBlock(dim, dim, 3, 1, 1,  pad_type=pad_type, norm="none", activation=activation)]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        residual = x
        output = self.model(x)
        output + residual
        return out

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



