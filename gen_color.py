from PIL import Image

size = (256, 256)
step = 25
save_dir_t = "./dataset/special/target_c"
save_dir_r = "./dataset/special/raw_ov"


def gen_color(r, g, b):
    while (r, g, b) != (0, 0, 0):
        img_t = Image.new("RGB", size=size, color=(r, g, b))
        img_r = img_t.convert("1")
        # 将灰度图像img_r转换为半色调图像
        img_t.save(f"{save_dir_t}/cc_{r:03d}{g:03d}{b:03d}.png")
        img_r.save(f"{save_dir_r}/cc_{r:03d}{g:03d}{b:03d}.png")
        b += step
        if b > 250:
            b = 0
            g += step
            if g > 250:
                g = 0
                r += step
                if r > 250:
                    r = 0
    return None


if __name__ == '__main__':
    img_t = Image.new("RGB", size=size, color=(0, 0, 0))
    img_r = img_t.convert("1")
    # 将灰度图像img_r转换为半色调图像
    img_t.save(f"{save_dir_t}/cc_{0:03d}{0:03d}{0:03d}.png")
    img_r.save(f"{save_dir_r}/cc_{0:03d}{0:03d}{0:03d}.png")
    gen_color(0, 0, 25)
