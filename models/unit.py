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


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()


    def forward(self, x):
        pass


class generateEndcoder(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, first_ch=64, norm_layer=nn.BatchNorm2d):
        down = []
        down.append(nn.Conv2d(input_nc, first_ch, kernel_size=4, stride=2, padding=2 ))
        
        

        for i in range(num_downs):
            down.append(i)


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



