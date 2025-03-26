import PIL.Image as Image
import pickle
import json
import time
import os
import numpy as np
from tqdm import tqdm
import pydicom
import cv2
from multiprocessing import Pool
from functools import partial
import ast

PAR_DIR = "<DIR_TO_DICOM_EMBED_DATASET>"
TAR_DIR = "<DIR_TO_OUTPUT_RESIZED_IMAGES>"


def get_otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def check_img_side(x):
    left = np.sum(x[:, 0])
    right = np.sum(x[:, -1])
    if left > right:
        return "L"
    else:
        return "R"


def resize_img(img, scale, intermethod=cv2.INTER_AREA, padding=False):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    # print("IMG SIZE:", size)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        # print(desireable_size)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=intermethod
    )  # this flips the desireable_size vector

    # Padding
    if padding:
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

    return resized_img


def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    orig_img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(orig_img)
    else:
        img = orig_img

    return (img,)


def pipe(path_list, image_size=1024):
    failed_list = []
    for path in tqdm(path_list):
        assert os.path.exists(path)
        if TAR_DIR is not None:
            dest = path.replace(PAR_DIR, TAR_DIR)
        else:
            dest = path
        dest = dest.replace(".dcm", "_resized.jpg")
        direct = os.path.dirname(dest)
        os.makedirs(direct, exist_ok=True)
        try:
            img = read_from_dicom(path, image_size)
            img.save(dest)
        except Exception as e:
            failed_list.append(path)
            print(f"Failed to process {path}")
            print(e)
            time.sleep(1)
            continue
        # break
    return failed_list


if __name__ == "__main__":
    pickle_list = "<PICKLE_FILE_OF_DICOM_IMAGES_TO_PROCESS>"

    NT = 12

    for pickle_file in pickle_list:
        img_paths = pickle.load(open(pickle_file, "rb"))

        print(f"Loaded {len(img_paths)} images from {pickle_file}")

        sub_img_paths = [
            {k: img_paths[k] for k in list(img_paths.keys())[i : len(img_paths) : NT]}
            for i in range(NT)
        ]
        for i in range(NT):
            print(
                f"Processing {len(sub_img_paths[i])} images in {pickle_file} with {NT} processes"
            )
            print(f"Writing to ./tmp/missing_idx.json")
        func = partial(pipe, image_size=1080)

        with Pool(NT) as p:
            results = p.map(func, sub_img_paths)
            p.close()
            p.join()
            with open(f"./tmp/missing_idx.json", "a") as fp:
                json.dump(results, fp)
