from PIL import Image
import os

base = "../img/"
out_sizes = [4, 8, 16, 32, 64]
in_folder = "128/00000/"

dirs = os.listdir(base + in_folder)

for size in out_sizes:
	out_folder = base + str(size) + "/00000/"
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)
	for item in dirs:
		if os.path.isfile(base + in_folder + item):
			im = Image.open(base + in_folder + item)
			imResize = im.resize((size, size), Image.BILINEAR)
			imResize.save(out_folder + item, "PNG")
			im.close()
			imResize.close()
