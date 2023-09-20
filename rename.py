from PIL import Image
import os

image_path = "./dataset1"
gray_path = "./dataset2"

count = 0

for filename in os.listdir(image_path):
    if count < 100:
        count += 1
        code = f'{count:05d}'  # 编码规则
        Image.open(f"{image_path}/{filename}").save(f"{image_path}/{code}.png")

count = 0
for filename in os.listdir(gray_path):
    if count < 100:
        count += 1
        code = f'{count:05d}'  # 编码规则
        Image.open(f"{gray_path}/{filename}").save(f"{gray_path}/{code}.png")