from PIL import Image
import os
import shutil
import numpy as np

base = "../img/"

def get_image(size):
        in_folder = base + str(size) + "/00000/"
        assert os.path.exists(in_folder)
        for item in os.listdir(in_folder):
                im = Image.open(in_folder + item)
                yield np.array(im, dtype=np.float32) / 127.5 - 1.0
                im.close()

def get_n_images(size, num):
        ret = []
        for i in get_image(size):
                ret.append(i)
                num = num - 1
                if num == 0:
                        break
        return len(ret), np.array(ret)

def save_image(size, name, image):
        out_folder = base + str(size) + "/gen/"
        if not os.path.exists(out_folder):
                os.mkdir(out_folder)
        im = Image.fromarray(((image + 1.0) * 127.5).astype(np.uint8))
        im.save(out_folder + str(name) + ".png", "PNG")
        im.close()

def clean(size):
        out_folder = base + str(size) + "/gen/"
        if os.path.exists(out_folder):
                shutil.rmtree(out_folder)
