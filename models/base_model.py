import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import pickle


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call  `BaseModel.__init__(self, opt)`
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        # save all the checkpoints to save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(self.save_dir)
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        # if opt.preprocess != 'scale_width':
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.ite_to_epoch_dict = dict()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(
                optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            print(load_suffix)
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    self.load_network(net, name, load_suffix, self.save_dir)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        """The paramerter is updated when step is called. """
        for scheduler in self.schedulers:
            scheduler.step()
        """The following is obtained as a example to display the learning rate"""
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda()
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1)

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        """pix2pixHDのほう"""
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {
                        k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print(
                        'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    import sys
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def old_load_networks(self, epoch):
        """
        もともとpix2pixの関数
        cycleでは使われている（かもしれない）がmulit pix2pixでは利用されていない．
        """
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                print(name)
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                print(net)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(
                    load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # need to copy keys here because we mutate in loop
                for key in list(state_dict.keys()):
                    print(key)
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def make_loss_hist_list(self):
        if self.opt.phase == "test" or not self.opt.continue_train:
            for name in self.loss_names:
                setattr(self, 'loss_' + name + '_it_hist', [])
                setattr(self, 'loss_' + name + '_epoch_last_hist', [])
                setattr(self, 'loss_' + name + '_during_epoch', [])
        else:
            for name in self.loss_names:

                it_name = name + '_it.pkl'
                loss_it_path = os.path.join(self.save_dir, it_name)
                with open(loss_it_path, "rb") as f:
                    it_loss_list = pickle.load(f)
                    setattr(self, 'loss_' + name + '_it_hist', it_loss_list)

                epoch_name = name + '_epoch_last.pkl'
                loss_epoch_path = os.path.join(self.save_dir, epoch_name)
                with open(loss_epoch_path, "rb") as f:
                    epoch_loss_list = pickle.load(f)
                    setattr(self, 'loss_' + name +
                            '_epoch_last_hist', epoch_loss_list)
                setattr(self, 'loss_' + name + '_during_epoch', [])

    def append_loss_during_epoch(self):
        for name in self.loss_names:
            tmplist = getattr(self, 'loss_' + name + '_during_epoch')
            # float(...) works for both scalar tensor and float number
            tmplist.append(float(getattr(self, 'loss_' + name)))

    def concat_and_save_loss_end_epoch(self, epoch):
        # 1エポックの終了時にduring_epochの内容をit_histに追加する
        for name in self.loss_names:
            loss_it_hist = getattr(self, 'loss_' + name + '_it_hist')
            loss_during_epoch = getattr(self, 'loss_' + name + '_during_epoch')
            self.add_first_and_last_iteration_index_to_dict(epoch, len(
                loss_it_hist), len(loss_it_hist) + len(loss_during_epoch) - 1)

            loss_it_hist += loss_during_epoch
            loss_epoch_last = getattr(
                self, 'loss_' + name + '_epoch_last_hist')
            loss_epoch_last.append(loss_it_hist[-1])
        self.save_loss_association()

    # def append_loss(self,is_last):
    #     for name in self.loss_names:
    #         if isinstance(name, str):
    #             if not is_last:
    #                 #tmplistが'loss_' + name + '_it_hist'
    #                 tmplist = getattr(self, 'loss_' + name + '_it_hist')
    #                 tmplist.append(float(getattr(self, 'loss_' + name)))  # float(...) works for both scalar tensor and float number
    #             else:
    #                 tmplist = getattr(self, 'loss_' + name + '_epoch_last_hist')
    #                 tmplist.append(float(getattr(self, 'loss_' + name)))

    def save_loss_association(self):

        for name in self.loss_names:
            tmplist = getattr(self, 'loss_' + name + '_it_hist')
            save_filename = '%s_it.pkl' % (name)
            save_path = os.path.join(self.save_dir, save_filename)
            with open(save_path, 'wb') as f:
                pickle.dump(tmplist, f)

            tmplist = getattr(self, 'loss_' + name + '_epoch_last_hist')
            save_filename = '%s_epoch_last.pkl' % (name)
            save_path = os.path.join(self.save_dir, save_filename)
            with open(save_path, 'wb') as f:
                pickle.dump(tmplist, f)
        save_path = os.path.join(self.save_dir, "ite_to_epoch.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.ite_to_epoch_dict, f)

    def add_first_and_last_iteration_index_to_dict(self, epoch, findex, lindex):
        self.ite_to_epoch_dict[epoch] = (findex, lindex)
