from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
#img1 = np.array(Image.open("./test_imgs/dog.png"))
#img2 = np.array(Image.open("./result_noise/restored_dog.jpg"))
img1 = np.array(Image.open("./test_imgs/dog.png"))
img2 = np.array(Image.open("./result_one/restored_dog.png"))
print(img1.shape)
print(img2.shape)

PSNR = psnr(img1, img2)
SSIM = ssim(img1, img2, channel_axis=2)

print("PSNR: ", PSNR)
print("SSIM: ", SSIM)

