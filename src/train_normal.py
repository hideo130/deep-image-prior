import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import utils
from torchsummary import summary
from torchvision import transforms
from PIL import Image

from models.dip import DIP
from util.print_loss import print_losses
from util.util import get_logger, get_noisy_img, make_mask
from util.make_video import make_video


@hydra.main(config_path='../configs/config.yaml')
def train(cfg):
    logger = get_logger("./log/")
    # lossutil = LossUtil("./log/")
    logger.info(cfg)
    ROOT = Path(utils.get_original_cwd()).parent
    print(ROOT)
    # data_path = ROOT.joinpath("datasets/img/1024-512.png")
    data_path = ROOT.joinpath("datasets/img/512-256.png")
    img = np.array(Image.open(data_path), dtype=float)
    img = img / 255

    result_dir = Path("./result_imgs/")
    if not result_dir.exists():
        Path(result_dir).mkdir(parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.image.type == "denoise":
        sigma = 0.1
        img = get_noisy_img(img, sigma)
        mask = None
        tmp = Image.fromarray(np.uint8(255*img))
        tmp.save(result_dir.joinpath("target.png"))
    elif "inpaint" in cfg.image.type:
        mask = make_mask(cfg, img)
        # ターゲット画像の保存
        tmp = mask*img
        tmp = Image.fromarray(np.uint8(255*tmp))
        tmp.save(result_dir.joinpath("target.png"))

        mask = torch.from_numpy(mask)
        mask = mask[None, :].permute(0, 3, 1, 2).to(device)
        logger.info(mask.shape)
    else:
        raise NotImplementedError(
            '[%s] is not Implemented' % cfg.image.type)

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = img[None, :].to(device)
    logger.info(img.shape)
    logger.info(img.dtype)

    np.random.seed(cfg.base_options.seed)
    torch.manual_seed(cfg.base_options.seed)
    torch.cuda.manual_seed_all(cfg.base_options.seed)
    torch.backends.cudnn.benchmark = False

    model = DIP(cfg, img, mask)
    logger.info(model.generator)

    if cfg.base_options.debugging:
        summary(model.generator, (1, 1024, 512))
        # summary(model.generator, (3, 256, 256))
        print('デバッグ中　途中で終了します!!!')
        exit(0)

    input_noise = torch.randn(1, 1, img.shape[2], img.shape[3], device=device)
    logger.info(input_noise.dtype)
    logger.info(input_noise.shape)
    for epoch in range(cfg.base_options.epochs):
        if epoch % cfg.base_options.print_freq == 0:
            epoch_start_time = time.time()

        if not cfg.base_options.do_fix_noise:
            input_noise = torch.randn(
                1, 1, img.shape[2], img.shape[3], dtype=torch.float64, device=device)
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
            new_img = new_img.permute(1, 2, 0).cpu().float().numpy()
            # print(new_img.shape, new_img.device, new_img.dtype)
            new_img = Image.fromarray(np.uint8(255*new_img))
            new_img.save(result_dir.joinpath("%05d.png" % epoch))

    model.save_networks("finish")
    model.save_losses()
    # 実験結果の動画
    freq = cfg.base_options.save_img_freq
    epochs = cfg.base_options.epochs
    logger.info("making result video")
    result_imgs = ["./result_imgs/%05d.png" %
                   (epoch) for epoch in range(epochs) if epoch % freq == 0]
    make_video("./", result_imgs, img.shape[3], img.shape[2])
    logger.info("making result video done!")


if __name__ == '__main__':
    train()
