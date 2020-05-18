import os
import pickle
from collections import OrderedDict

import torch

from .networks import define_G


class DIP():
    def __init__(self, cfg, raw_img, mask=None):
        models = cfg.models
        self.loss_names = ["L2_loss"]
        self.L2_loss_hist = []
        self.save_dir = "./"
        self.gpu_ids = cfg.base_options.gpu_ids
        self.generator = define_G(models.input_nc, cfg.image.output_nc, models.ngf, models.model_name, models.norm,
                                  use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
        self.model_names = ["generator"]
        self.criterion = torch.nn.MSELoss()
        self.fix_noise = True
        self.optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=cfg.optimizer.lr, betas=(0.5, 0.999))
        self.raw_img = raw_img
        self.mask = mask

    def forward(self, input_noize):
        """
        input_noiseはforward内で生成するか，引数にするか
        """
        self.gimg = self.generator(input_noize)

    def backward(self):
        if self.mask is not None:
            self.L2_loss = self.criterion(self.gimg*self.mask, self.raw_img)
        else:
            self.L2_loss = self.criterion(self.gimg, self.raw_img)
        self.L2_loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.append_losses()

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            save_filename = '%s_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda()
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def append_losses(self):
        for name in self.loss_names:
            tmplist = getattr(self, name + '_hist')
            tmplist.append(float(getattr(self, name)))

    def save_losses(self):
        for name in self.loss_names:
            tmplist = getattr(self, name + '_hist')
            save_filename = '%s.pkl' % (name)
            save_path = os.path.join(self.save_dir, save_filename)
            with open(save_path, 'wb') as f:
                pickle.dump(tmplist, f)

    def get_current_losses(self):
        losses = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = float(getattr(self, name))
        return losses

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        """The paramerter is updated when step is called. """
        for scheduler in self.schedulers:
            scheduler.step()
        """The following is obtained as a example to display the learning rate"""
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self,  name)
                net.load_state_dict(torch.load(load_path))
