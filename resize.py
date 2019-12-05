from PIL import Image
import os

base = "../img/"
out_sizes = [4, 8, 16, 32, 64]
in_size = 128
num_sets = 5

for n in range(num_sets):
        in_folder = os.path.join(base, str(in_size), str(n * 1000).rjust(5, "0"))
        if os.path.exists(in_folder) and os.path.isdir(in_folder):
                for size in out_sizes:
                        out_folder = os.path.join(base, str(size))
                        if not os.path.exists(out_folder):
                                os.mkdir(out_folder)
                        out_folder = os.path.join(out_folder, str(n * 1000).rjust(5, "0"))
                        if not os.path.exists(out_folder):
                                os.mkdir(out_folder)
                        dirs = os.listdir(in_folder)
                        for item in dirs:
                                if os.path.isfile(os.path.join(in_folder, item)):
                                        im = Image.open(os.path.join(in_folder, item))
                                        imResize = im.resize((size, size), Image.BILINEAR)
                                        imResize.save(os.path.join(out_folder, item), "PNG")
                                        im.close()
                                        imResize.close()
        else:
               print("Folder not found:", in_folder)