from .base_model import BaseModel
from .unit import UNIT
from . import networks
import torch
import itertools
from apex import amp

class UnitModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ["recon_A", "recon_B", "kl_A", "kl_B", "recon_kl_AtoB",
                           "recon_kl_BtoA", "cycle_A", "cycle_B", "D_A", "D_B", "G_A", "G_B"]
        self.make_loss_hist_list()
        self.model_names = ["Unit", "D_A", "D_B"]
        """
        set save image names
        """
        visual_names_A = ['real_A', 'cross_AtoB', 'cycle_A',"vae_recon_A"]
        visual_names_B = ['real_B', 'cross_BtoA', 'cycle_B',"vae_recon_B"]
        if self.opt.dataset_mode != 'single':
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        else:
            self.visual_names = visual_names_A

        '''
        unit init:input_dim, dim, num_down, norm, activation, num_resblock, output_dim, pad_type
        '''
        self.netUnit = UNIT(3, 128, 2, "bn", "tanh", 3, 3, "zero",)
        if self.isTrain:
            networks.init_net(self.netUnit, opt.init_type, opt.init_gain, self.gpu_ids)
            self.use_sigmoid = False if self.opt.gan_mode == "lsgan" else True
            getIntermFeat = False
            # opt.num_D = 1
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D,
                                          opt.norm, self.use_sigmoid, opt.num_D, getIntermFeat, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D,
                                          opt.norm, self.use_sigmoid, opt.num_D, getIntermFeat, self.gpu_ids)

        if self.isTrain:
            self.critericonRecon = torch.nn.L1Loss()
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # print(self.criterionGAN.loss)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netUnit.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(
            ), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            use_FP16 = False
            if use_FP16:
                self.netUnit, self.optimizer_G = amp.initialize(self.netUnit, self.optimizer_G, opt_level="01")
                D_list, self.optimizer_G = amp.initialize([self.netD_A, self.netD_B], self.optimizer_D, opt_level="01")
                self.netD_A, self.netD_B = D_list[0], D_list[1]
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
            # print(self.real_A.size())
            self.h_A, noise_A = self.netUnit.encode_a(self.real_A)
            self.h_B, noise_B = self.netUnit.encode_b(self.real_B)
            # print(self.h_A.size())
            # print(noise_A.size())

            # vae

            self.vae_recon_A = self.netUnit.decode_a(self.h_A + noise_A)
            self.vae_recon_B = self.netUnit.decode_b(self.h_B + noise_B)
            # cross domain decode
            self.cross_BtoA = self.netUnit.decode_a(self.h_B + noise_B)
            self.cross_AtoB = self.netUnit.decode_b(self.h_A + noise_A)
            # encode again
            # print(self.cross_AtoB.size())
            # print(self.cross_AtoB)
            self.recon_h_BtoA, noise_A_recon = self.netUnit.encode_a(self.cross_BtoA)
            self.recon_h_AtoB, noise_B_recon = self.netUnit.encode_b(self.cross_AtoB)
            # print(self.recon_h_AtoB.size())
            # decode again
            self.cycle_A = self.netUnit.decode_a(self.recon_h_AtoB + noise_B_recon)
            self.cycle_B = self.netUnit.decode_b(self.recon_h_BtoA + noise_A_recon)
        else:
            pass

    def backward_G(self):
        # vae
        self.loss_recon_A = self.critericonRecon(self.real_A, self.vae_recon_A)
        self.loss_recon_B = self.critericonRecon(self.real_B, self.vae_recon_B)
        self.loss_kl_A = self.__compute_kl(self.h_A)
        self.loss_kl_B = self.__compute_kl(self.h_B)

        self.loss_recon_kl_BtoA = self.__compute_kl(self.recon_h_BtoA)
        self.loss_recon_kl_AtoB = self.__compute_kl(self.recon_h_AtoB)

        # cycle conin
        self.loss_cycle_A = self.critericonRecon(self.real_A, self.cycle_A)
        self.loss_cycle_B = self.critericonRecon(self.real_B, self.cycle_B)

        # GAN lossG_A(G_A(A))
        # tmp = self.netD_A(self.cross_BtoA)
        # for i in tmp:
        #     print("type", type(i[0]))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.cross_BtoA), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.cross_AtoB), True)
        # print("loss_G_A", self.loss_G_A.size())
        # print("self.loss_G_A", self.loss_G_A)
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_kl_A + self.loss_kl_B + self.loss_recon_A + self.loss_recon_B + \
            self.loss_cycle_A + self.loss_cycle_B + \
            self.loss_recon_kl_AtoB + self.loss_recon_kl_BtoA

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
            self.netD_A, self.real_B, self.cross_AtoB.detach())

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.real_A, self.cross_BtoA.detach())

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
        # print(pred_real)
        # print(loss_D_real.size())
        # print("loss_D_real",loss_D_real)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
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

    def save(self, which_epoch):
        self.save_network(self.netUnit, 'Unit', which_epoch, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', which_epoch, self.gpu_ids)
        self.save_network(self.netD_A, 'D_B', which_epoch, self.gpu_ids)
