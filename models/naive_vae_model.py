from .base_model import BaseModel
# from . import networks
from . import unit
import torch.nn as nn
import torch


class NaiveVaeModel(BaseModel):
    # def modify_commandline_options(parser, is_train=True):
    #     if is_train:
    #         parser.add_argument("--lambda_KL", type=float, default=1, help="weight of KL Divergence")

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        self.opt = opt
        self.loss_names = ["loss_KL", "loss_BCE"]
        # This is called in save_networks
        self.model_names = ["VAE"]
        self.netVAE = unit.naive_VAE()
        self.netVAE.cuda(self.opt.gpu_ids[0])
        print(self.netVAE)
        # This is called when 
        self.visual_names = ["input"]

        if self.isTrain:
            self.optimizers = []
            self.optimizer = torch.optim.Adam(self.netVAE.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer)
            self.criterionBCE = nn.BCELoss(reduction="mean")

    def set_input(self, input):
        self.input = input.to(self.device)

    def forward(self):
        self.recon_x, self.mu, self.logvar = self.netVAE(self.input)

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def backward(self):
        self.loss_BCE = self.criterionBCE(self.recon_x, self.input)
        self.loss_KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        self.loss_VAE = self.loss_BCE + self.opt.lambda_KLD * self.loss_KLD
        self.loss_VAE.backward()

    # def save_networks(self,epoch)

def loss_function(recon_x, x, mu, logvar):
    # 古賀くんは画像サイズ128×128で0.001でうまくいかなくて0.0001でうまくいったらしい
    # lambda_KL = 0.01
    BCE = nn.BCELoss(recon_x, x.view(-1, 784), reduction="mean")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
