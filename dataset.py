from PIL import Image
import os
import numpy as np

base = "../00000/"

def get_image(size):
        in_folder = base + str(size) + "/"
        assert os.path.exists(in_folder)
        for item in os.listdir(in_folder):
                im = Image.open(in_folder + item)
                yield np.array(im)
                im.close()

def get_n_images(size, num):
        ret = []
        for i in get_image(size):
                ret.append(i)
                num = num - 1
                if num == 0:
                        break
        return np.array(ret)
