import cv2
import numpy as np
import random
from PIL import Image
from skimage.util import random_noise

color_img = cv2.imread('I0.jpg')
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('I1.png', gray_img)

image = Image.open("I1.png")
image = np.asarray(image)
noise_img1 = random_noise(image, mode='speckle', var=0.05)
noise_img1 = (255 * noise_img1).astype(np.uint8)
cv2.imwrite('noise_1.png', noise_img1)
noise_img2 = random_noise(image, mode='gaussian', var=0.01)
noise_img2 = (255 * noise_img2).astype(np.uint8)
cv2.imwrite('noise_2.png', noise_img2)
noise_img3 = random_noise(image, mode='s&p')
noise_img3 = (255 * noise_img3).astype(np.uint8)
cv2.imwrite('noise_3.png', noise_img3)

kernel_size = (2, 2)
filtered_image1 = cv2.blur(noise_img1, kernel_size)
filtered_image2 = cv2.blur(noise_img2, kernel_size)
filtered_image3 = cv2.blur(noise_img3, kernel_size)
cv2.imwrite('blur1.png', filtered_image1)
cv2.imwrite('blur2.png', filtered_image2)
cv2.imwrite('blur3.png', filtered_image3)

kernel_size = (3, 3)
gaussian_image1 = cv2.GaussianBlur(noise_img1, kernel_size, 0)
gaussian_image2 = cv2.GaussianBlur(noise_img2, kernel_size, 0)
gaussian_image3 = cv2.GaussianBlur(noise_img3, kernel_size, 0)
cv2.imwrite('gaussian1.png', gaussian_image1)
cv2.imwrite('gaussian2.png', gaussian_image2)
cv2.imwrite('gaussian3.png', gaussian_image3)

median_image1 = cv2.medianBlur(noise_img1, 3)
median_image2 = cv2.medianBlur(noise_img2, 3)
median_image3 = cv2.medianBlur(noise_img3, 3)
cv2.imwrite('median1.png', median_image1)
cv2.imwrite('median2.png', median_image2)
cv2.imwrite('median3.png', median_image3)

img = cv2.imread('I0.jpg')
# 产生高斯随机数
rayleigh = np.random.rayleigh(70.0, size=img.shape)
rayleigh_img = img + rayleigh
rayleigh_img = np.clip(rayleigh_img, 0, 255)
cv2.imwrite("color_rayleigh.png", rayleigh_img)
rayleigh_img = np.uint8(rayleigh_img)
rayleigh_filtered_img = cv2.medianBlur(rayleigh_img, 3)
cv2.imwrite("rayleigh.png", rayleigh_filtered_img)

# 转化成向量
x = img.reshape(1, -1)
# 设置信噪比
SNR = 0.85
# 得到要加噪的像素数目
noise_num = x.size * (1 - SNR)
# 得到需要加噪的像素值的位置
list = random.sample(range(0, x.size), int(noise_num))
for i in list:
    if random.random() >= 0.5:
        x[0][i] = 0
    else:
        x[0][i] = 255
salt_and_pepper_img = x.reshape(img.shape)
cv2.imwrite("color_s&p.png", salt_and_pepper_img)
salt_and_pepper_img = np.uint8(salt_and_pepper_img)
salt_and_pepper_filtered_img = cv2.medianBlur(salt_and_pepper_img, 3)
cv2.imwrite("s&p.png", salt_and_pepper_filtered_img)
