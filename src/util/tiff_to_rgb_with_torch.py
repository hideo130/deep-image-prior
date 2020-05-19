import torch
import numpy as np
from PIL import Image


def load_illuminantA(name="A.csv"):
    sd_light_source = np.loadtxt(name, skiprows=1, dtype="float")
    sd_light_source = sd_light_source[np.where(sd_light_source[:, 0] >= 400)]
    sd_light_source = sd_light_source[:, 1:2]
    sd_light_source = sd_light_source[::2]
    sd_light_source = sd_light_source[:44]
    # sd_light_source = sd_light_source / np.max(sd_light_source)
    return sd_light_source


class Tiff2rgb():
    def __init__(self, dist_path, cmf_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cmf = np.loadtxt(cmf_path, delimiter=",")

        # HSIが400nm以上のため等色関数も400nm以上のみを利用
        cmf = cmf[np.where(cmf[:, 0] >= 400)]
        # 光源の分光分布の5nm刻みをHSIと同じ10nm刻みに変更
        cmf = cmf[::2]
        cmf = cmf[:44, :]

        sd_light_source = load_illuminantA(name=dist_path)
        sd_light_source = sd_light_source.astype(np.float32)
        sd_light_source = torch.from_numpy(sd_light_source).to(self.device)
        # print(sd_light_source.dtype)
        cmf = cmf[:, 1:].astype(np.float32)
        cmf = torch.from_numpy(cmf).to(self.device)
        # print(cmf.dtype)

        self.nmf_multi_ld = cmf * sd_light_source
        self.flag_const_100 = True
        y = self.nmf_multi_ld[:, 1]
        if self.flag_const_100:
            self.k = 100 / torch.sum(y)
        else:
            self.k = 1 / torch.sum(y)

    def tiff_to_rgb(self, himg):
        """
        input: ハイパースペクトル画像　HSI（numpy型）

        return: RGB画像（Image objedct）
        """

        XYZ = torch.einsum("whc, cm -> whm", himg, self.nmf_multi_ld)
        XYZ = XYZ * self.k
        xyz_to_srgb = torch.tensor([[3.2406255, -1.537208, -0.4986286],
                                    [-0.9689307, 1.8757561, 0.0415175],
                                    [0.0557101, -0.2040211, 1.0569959]], device=self.device)

        rgb_img = torch.einsum("whc, mc -> whm", XYZ, xyz_to_srgb)

        rgb_img = torch.where(rgb_img >= 0, rgb_img, torch.tensor([0.], device=self.device))
        if self.flag_const_100:
            # HSI画像配布元と同じガンマ補正（ガンマ=0.6）をしている
            rgb_img = torch.pow(rgb_img/255, 0.6) * 255
        else:
            # XYZからsRGBへのレンダリングに乗っているガンマ補正
            rgb_img = torch.where(rgb_img <= 0.0031308, 12.92 *
                                  rgb_img, 1.055 * torch.pow(rgb_img, 1/2.4) - 0.055)

        if self.flag_const_100:
            img = Image.fromarray(np.uint8(rgb_img.cpu().float().numpy()))
        else:
            img = Image.fromarray(np.uint8(255*rgb_img.cpu().float().numpy()))
        return img
