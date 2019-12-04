from PIL import Image
import os
import shutil
import numpy as np

base = "../img/"
num_sets = 5

def get_image(size):
        if get_image.item == 999:
                get_image.n += 1
                get_image.item = 0
        else:
                get_image.item += 1
        if get_image.n == num_sets:
                get_image.n = 0
        in_folder = os.path.join(base, str(size), str(get_image.n * 1000).rjust(5, '0'))
        if os.path.exists(in_folder):
                filename = os.path.join(in_folder, str(get_image.n * 1000 + get_image.item).rjust(5, "0") + ".png")
                if os.path.exists(filename):
                        im = Image.open(filename)
                        val = np.array(im, dtype=np.float32) / 127.5 - 1.0
                        im.close()
                        return val
get_image.n = 0
get_image.item = 0

def get_n_images(size, num):
        val = [get_image(size) for i in range(num)]
        return len(val), val

def save_image(size, name, image):
        out_folder = base + str(size)
        if not os.path.exists(out_folder):
                os.mkdir(out_folder)
        out_folder +=  "/gen/"
        if not os.path.exists(out_folder):
                os.mkdir(out_folder)
        im = Image.fromarray(((image + 1.0) * 127.5).astype(np.uint8))
        im.save(out_folder + str(name) + ".png", "PNG")
        im.close()

def clean(size):
        out_folder = base + str(size) + "/gen/"
        if os.path.exists(out_folder):
                shutil.rmtree(out_folder)
