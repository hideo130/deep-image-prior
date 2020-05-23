import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra import utils
from skimage.external.tifffile import imread
from torchsummary import summary
from torchvision import transforms

from models.dip import DIP
from util.print_loss import print_losses
from util.tiff_to_rgb_with_torch import Tiff2rgb
from util.util import get_logger, get_noisy_img, make_mask_himg
from util.make_video import make_video


@hydra.main(config_path='../configs/config.yaml')
def train(cfg):
    logger = get_logger("./log/")
    # lossutil = LossUtil("./log/")
    logger.info(cfg)
    ROOT = Path(utils.get_original_cwd()).parent
    print(ROOT)
    data_path = ROOT.joinpath("datasets/hsimg/data.tiff")
    himg = imread(str(data_path))
    himg = himg / 2**16
    # 分光画像の範囲を光源の範囲に制限
    himg = himg[:, :, :44]
    himg = np.where(himg < 0, 0, himg)
    nhimg = np.empty((512, 512, himg.shape[2]))
    for i in range(himg.shape[2]):
        logger.info("resize %d channel..." % i)
        ch = himg[:, :, i]
        nhimg[:, :, i] = cv2.resize(
            255 * ch, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        logger.info("resize  %d channel... done!" % i)
    himg = nhimg / 255
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.image.type == "denoise":
        sigma = 0.1
        himg = get_noisy_img(himg, sigma)
        mask = None
    elif "inpaint" in cfg.image.type:
        mask = make_mask_himg(cfg, himg)
        logger.info(mask.dtype)
        mask = torch.from_numpy(mask)
        mask = mask[None, :].permute(0, 3, 1, 2).to(device)
        logger.info(mask.shape)
        logger.info(mask.dtype)

    else:
        raise NotImplementedError(
            '[%s] is not Implemented' % cfg.image.type)

    transform = transforms.Compose([transforms.ToTensor()])
    himg = transform(himg)
    himg = himg[None, :].float().to(device)

    dist_path = ROOT.joinpath("datasets/csvs/D65.csv")
    cmf_path = ROOT.joinpath("datasets/csvs/CIE1964-10deg-XYZ.csv")
    tiff2rgb = Tiff2rgb(dist_path, cmf_path)

    # ターゲット画像を保存
    if cfg.image.type == "denoise":
        img = tiff2rgb.tiff_to_rgb(himg[0].permute(1, 2, 0))
    else:
        tmp = himg * mask
        logger.info(tmp.shape)
        logger.info(tmp.dtype)
        img = tiff2rgb.tiff_to_rgb(tmp[0].permute(1, 2, 0))
    result_dir = Path("./result_imgs/")
    if not result_dir.exists():
        Path(result_dir).mkdir(parents=True)
    img.save(result_dir.joinpath("target.png"))

    logger.info(himg.shape)
    logger.info(himg.dtype)
    np.random.seed(cfg.base_options.seed)
    torch.manual_seed(cfg.base_options.seed)
    torch.cuda.manual_seed_all(cfg.base_options.seed)
    torch.backends.cudnn.benchmark = False

    model = DIP(cfg, himg, mask)
    logger.info(model.generator)

    if cfg.base_options.debugging:
        summary(model.generator, (1, 512, 512))
        # summary(model.generator, (3, 256, 256))
        print('デバッグ中　途中で終了します!!!')
        exit(0)

    input_noise = torch.randn(
        1, 1, himg.shape[2], himg.shape[3], device=device)
    logger.info(input_noise.dtype)
    for epoch in range(cfg.base_options.epochs):
        if epoch % cfg.base_options.print_freq == 0:
            epoch_start_time = time.time()

        if not cfg.base_options.do_fix_noise:
            input_noise = torch.randn(
                1, 1, himg.shape[2], himg.shape[3], dtype=torch.float32, device=device)
        model.forward(input_noise)
        # logger.info(model.gimg.shape)
        # logger.info(model.gimg.dtype)
        model.optimize_parameters()
        # model.update_learning_rate()
        losses = model.get_current_losses()

        if epoch % cfg.base_options.print_freq == 0:
            num = cfg.base_options.print_freq
            t1 = (time.time() - epoch_start_time)
            t2 = float(t1) / num
            print("%dエポックの所要時間%.3f 平均時間%.3f" % (num, t1, t2))
            print_losses(epoch, losses)

        if epoch % cfg.base_options.save_model_freq == 0:
            model.save_networks(epoch)
            model.save_losses()
        if epoch % cfg.base_options.save_img_freq == 0:
            new_img = model.gimg.detach()[0]
            # CHW -> HWC
            new_img = new_img.permute(1, 2, 0)
            # print(new_img.shape, new_img.device, new_img.dtype)
            img = tiff2rgb.tiff_to_rgb(new_img)
            img.save(result_dir.joinpath("%05d.png" % epoch))

    model.save_networks("finish")
    model.save_losses()
    # 実験結果の動画
    freq = cfg.base_options.save_img_freq
    epochs = cfg.base_options.epochs
    result_imgs = ["./result_imgs/%05d.png" %
                   (epoch) for epoch in range(epochs) if epoch % freq == 0]
    make_video("./", result_imgs, width=himg.shape[3], height=himg.shape[3])


if __name__ == '__main__':
    train()
