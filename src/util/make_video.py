import cv2
from pathlib import Path


def make_video(save_dir, img_names, IMG_SIZE=512):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save_name = Path(save_dir).joinpath('result_video.mp4')
    video = cv2.VideoWriter(str(save_name), fourcc, 20.0, (IMG_SIZE, IMG_SIZE))
    for img_file in img_names:
        img = cv2.imread(img_file)
        video.write(img)
    video.release()


if __name__ == "__main__":
    img_dir = "./"
    img_names = [Path(img_dir).joinpath("%d.png" % (10*i))
                 for i in range(200)]
    make_video(img_dir, img_names)
