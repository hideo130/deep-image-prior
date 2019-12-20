import torch
from torch.utils.data import Dataset
from data.image_folder import make_dataset
from PIL import Image
import os
from torchvision import transforms
import random


class Transforms(object):
    def __init__(self):
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
        # self.frip = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomV])
        self.transform = transforms.Compose(transform_list)

    def __call__(self, A, B):
        if random.random() < 0.5:
            A = A.transpose(Image.FLIP_LEFT_RIGHT)
            B = B.transpose(Image.FLIP_LEFT_RIGHT)
        # call standard transformation function
        if random.random() < 0.5:
            A = A.transpose(Image.FLIP_TOP_BOTTOM)
            B = B.transpose(Image.FLIP_TOP_BOTTOM)
        A = self.transform(A)
        B = self.transform(B)
        return A, B


class Aligned_dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.transforms = Transforms()
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # self.dir_AB = "../../cycle-gan/datasets/L159/original/CK19andHE/256/train/"
        # max_dataset_size = float("inf")
        self.AB_paths = sorted(make_dataset(
            self.dir_AB, opt.max_dataset_size))  # get image paths
        self.data_num = len(self.AB_paths)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        AB_path = self.AB_paths[idx]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        if self.transforms:
            A, B = self.transforms(A, B)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}


def main():
    from tqdm import tqdm
    from torch.utils.data.dataloader import DataLoader

    # path = 'data/processed/ver2.2/train_dataset.hdf5'
    # dataset = GNSSDatasets(path, 1200)
    aligned_dataset = Aligned_dataset([])
    dataloader = DataLoader(
        aligned_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4
    )
    for data in tqdm(dataloader):
        print(data["A"].size())
        break
# main()
