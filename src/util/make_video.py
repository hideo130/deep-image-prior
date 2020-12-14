from typing import List, Union
import cv2
from pathlib import Path, PosixPath


def make_video(
    save_dir: str, img_names: List[Union[str, PosixPath]], width=512, height=512
):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    save_name = Path(save_dir).joinpath("result_video.mp4")
    video = cv2.VideoWriter(str(save_name), fourcc, 20.0, (width, height))
    for img_file in img_names:
        img = cv2.imread(str(img_file))
        video.write(img)
    video.release()


if __name__ == "__main__":
    img_dir = "./"
    img_names = [Path(img_dir).joinpath("%d.png" % (10 * i)) for i in range(200)]
    make_video(img_dir, img_names)
