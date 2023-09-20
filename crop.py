from PIL import Image
import os


def crop_image(image_path, output_path, gray_path, crop_size, code):
    im = Image.open(image_path)
    width, height = im.size  # Get dimensions

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    cropImgPath = os.path.join(output_path, code + ".png")
    im.save(cropImgPath)
    Image.open(cropImgPath).convert("1").save(os.path.join(gray_path, code + ".png"))

image_path = "./JPEGImages"
output_path = "./dataset1"
gray_path = "./dataset2"
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(gray_path):
    os.makedirs(gray_path)
crop_size = 256

count = 0

for filename in os.listdir(image_path):
    name, suffix = filename.split('.')
    if not filename.endswith(".png"):
        Image.open(f"./JPEGImages/{filename}").save(f"./JPEGImages/{name}.png")
        os.remove(f"./JPEGImages/{filename}")
        #  Image.open(f"./JPEGImages/{name}.png").convert("RGBA").save(f"./JPEGImages/{name}.png")
    count += 1
    code = f'{count:05d}'  # 编码规则
    image_file = os.path.join(image_path, f"{name}.png")
    crop_image(image_file, output_path, gray_path, crop_size, code)
