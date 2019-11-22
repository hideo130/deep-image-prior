from skimage.measure import compare_ssim, compare_psnr
import cv2
import time

def measurement(func, **kwargs):
    start = time.time()
    val = func(kwargs["img1"], kwargs["img2"] )
    end = time.time()
    return val, end-start

if __name__ == "__main__":

    img1 = cv2.imread("epoch198_fake_B.png")
    img2 = cv2.imread("epoch198_real_B.png")
    # img2 = cv2.imread("epoch198_real_A.png")
    print("psnr: %f, time: %lf[sec]" % measurement(compare_psnr, img1=img1, img2=img2,multichannel=True))
    # print("ssim: %f, time: %lf[sec]" % measurement(compare_ssim, img1=img1, img2=img2,multichannel=True ))
    
    val = compare_ssim(img1, img2,multichannel=True)
    print(val)