from .base_model import BaseModel
from .unit import UNIT
from . import networks
import torch
import itertools


class UnitModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ["recon_A", "recon_B", "kl_A", "kl_B", "recon_kl_A",
                           "recon_kl_B", "cycle_A", "cycle_B", "D_A", "D_B", "G_A", "G_B"]
        self.make_loss_hist_list()

        self.unit = UNIT()
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            networks.init_net(self.unit)
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.critericonRecon = torch.nn.L1loss()

        if self.isTrain:
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.unit.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(
            ), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
            self.h_A, noise_A = self.unit.enc_a(self.real_A)
            self.h_B, noise_B = self.unit.enc_b(self.real_B)
            # vae
            self.recon_A = self.unit.dec_a(self.h_A + noise_A)
            self.recon_B = self.unit.dec_b(self.h_B + noise_B)
            # cross domain decode
            self.fake_A = self.unit.dec_a(self.h_B + noise_A)
            self.fake_B = self.unit.enc_b(self.h_A + noise_B)
            # encode again
            self.recon_h_A, _ = self.unit.enc_a(self.fake_B)
            self.recon_h_B, _ = self.unit.enc_b(self.fake_A)

            # decode again
            self.cycle_A = self.unit.dec_a(self.recon_h_A + noise_A)
            self.cycle_B = self.unit.dec_b(self.recon_h_B + noise_B)
        else:
            pass

    def backward_D_basic(self, netD):
        pass

    def backward_G(self):
        # vae
        self.loss_recon_A = self.critericonRecon(self.real_A, self.recon_A)
        self.loss_recon_B = self.critericonRecon(self.real_B, self.recon_B)
        self.loss_kl_A = self.__compute_kl(self.h_A)
        self.loss_kl_B = self.__compute_kl(self.h_B)

        self.loss_recon_kl_A = self.__compute_kl(self.recon_h_A)
        self.loss_recon_kl_B = self.__compute_kl(self.recon_h_B)

        # cycle conin
        self.loss_cycle_A = self.critericonRecon(self.real_A, self.cycle_A)
        self.loss_cycle_B = self.critericonRecon(self.real_B, self.cycle_B)

        # GAN lossG_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_kl_A + self.loss_kl_B + self.loss_recon_A + self.loss_recon_B \
            + self.loss_cycle_A + self.loss_cycle_B + \
            self.loss_recon_kl_A + self.loss_recon_kl_B

        self.loss_G.backward()


    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.real_A, self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # update D parameters
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()
