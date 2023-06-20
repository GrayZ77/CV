import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

color_img = cv2.imread('I0.jpeg')
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('I1.png', gray_img)

img = cv2.imread("nju.jpg")
img = cv2.resize(img, (5922, 3567))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite('I2.png', thresh)

img = Image.open('I1.png')
img_nju = Image.open('I2.png')
width, height = img.size
img_array = img.load()
double_array = img_nju.load()
white = 0b00000001
black = 0b11111110
cnt = 1
for i in range(8):
    for x in tqdm(range(width)):
        for y in range(height):
            gray = img_array[x, y]
            double = double_array[x, y]
            if double == 255:
                new_gray = gray | white
            else:
                new_gray = gray & black
            img_array[x, y] = new_gray
    white = (white << 1) & 0xFF
    black = ((black << 1) | 1) & 0xFF
    filename = f'result_gray/result{cnt}.png'
    img.save(filename)
    cnt += 1

img_color = Image.open('I0.jpeg')
r, g, b = img_color.split()
r_array = np.array(r)
g_array = np.array(g)
b_array = np.array(b)
double_array = np.array(img_nju)
white = 0b00000001
black = 0b11111110
cnt = 1
for i in range(8):
    for x in tqdm(range(width)):
        for y in range(height):
            double = double_array[y, x]
            if double == 255:
                r_array[y, x] |= white
                g_array[y, x] |= white
                b_array[y, x] |= white
            else:
                r_array[y, x] &= black
                g_array[y, x] &= black
                b_array[y, x] &= black
    img_result = Image.fromarray(np.dstack((r_array, g_array, b_array)))
    white = (white << 1) & 0xFF
    black = ((black << 1) | 1) & 0xFF
    filename = f'result_color/result{cnt}.png'
    img_result.save(filename)
    cnt += 1

