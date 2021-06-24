import os
import shutil

from torchvision import utils

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def save_images(images, output_dir, file_prefix, nrows, iteration):
    utils.save_image(
        images,
        os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.png"),
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )