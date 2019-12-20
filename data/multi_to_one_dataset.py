import os.path
import random
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class Multi_to_oneDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # we manually crop and flip in __getitem__ to make sure we apply the same crop and flip for image A and B
        # we disable the cropping and flipping in the function get_transform
        self.transform_A = get_transform(opt, grayscale=(input_nc == 1), crop=False, flip=False)
        self.transform_B = get_transform(opt, grayscale=(output_nc == 1), crop=False, flip=False)
        # set crop position
        AB_path = self.AB_paths[0]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w3 = int(w / 3)
        if self.opt.MTorCK == "MT":
            self.start = w3
        elif self.opt.MTorCK == 'CK':
            self.start = 2*w3
        else:
            assert('You input %s. You have to input MT or CK.'%(self.opt.MTorCK))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B,C, A_paths and B_paths,C_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            C (tensor) - - its corresponding image in the other target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            C_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w3 = int(w / 3)
        if not self.opt.no_resize:
            A = AB.crop((0, 0, w3, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
            B = AB.crop((w3, 0, w, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        else:
            # print("hoge")
            A = AB.crop((0, 0, w3, h))
            B = AB.crop((self.start, 0, self.start + w3, h))
        # apply the same cropping to both A and B
        if 'crop' in self.opt.preprocess:
            x, y, h, w = transforms.RandomCrop.get_params(A, output_size=[self.opt.crop_size, self.opt.crop_size])
            A = A.crop((x, y, w, h))
            B = B.crop((x, y, w, h))
        # apply the same flipping to both A and B
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = A.transpose(Image.FLIP_LEFT_RIGHT)
            B = B.transpose(Image.FLIP_LEFT_RIGHT)
        if (not self.opt.no_horizon_flip) and random.random()<0.5:
            A = A.transpose(Image.FLIP_TOP_BOTTOM)
            B = B.transpose(Image.FLIP_TOP_BOTTOM)
        # call standard transformation function
        A = self.transform_A(A)
        B = self.transform_B(B)
        
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)