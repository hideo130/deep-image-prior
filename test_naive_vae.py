import models.unit as unit
from options.train_options import TrainOptions
import pathlib
from torchvision import datasets, transforms
import torch
import argparse
from models.naive_vae_model import NaiveVaeModel
# 便利そう　あとで調べる-1の理由は？
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

dataset_train = datasets.MNIST(
    './mnist',
    train=True,
    download=True,
    transform=transform)

dataset_test = datasets.MNIST(
    "./mnist",
    train=False,
    download=True,
    transform=transform
)

data_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=10,
    shuffle=True,
    num_workers=4
)

if __name__ == "__main__":
    # opt = TrainOptions().parse()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--lambda_KLD", type=float, default=1.00)
    parser.add_argument("--isTrain", action="store_true")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--name", type=str, default="VAE")
    parser.add_argument("--lr", type=float, default=0.0002)
    
    parser.set_defaults(isTrain=True)

    opt = parser.parse_args()
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    model = NaiveVaeModel(opt)
     
    dataset_size = len(dataset_train)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    for epoch in range(100):
        # losses = []
        for i, data in enumerate(data_loader):
            data = data[0]
            data = data.to("cuda")
            model.set_input(data)
            model.optimize_parameters()
            # model.append_loss_during_epoch()
            if i % 100 == 0:
                print("epoch:%3d, ite:%d curennt loss is %.3f" % (epoch, i, model.loss_VAE))

        # model.concat_and_save_loss_end_epoch(epoch)
        # model.update_learning_rate()
        if (epoch + 1) % 20 == 0: 
            model.save_networks(epoch)


# def train(epoch):
#     model.eval()
