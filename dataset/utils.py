import cv2
import numpy as np
import pydicom
from PIL import Image
from transformers import BertTokenizer, AutoTokenizer
from sklearn.metrics import roc_curve
from .constants_val import *


def read_from_dicom(img_path, imsize=None, transform=None, return_orig_img=False):
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

    if return_orig_img:
        return img, orig_img
    else:
        return img


def resize_img(img, scale, intermethod=cv2.INTER_AREA, rois=None):
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
    resized_rois = []

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        if rois is not None:
            for roi in rois:
                if len(roi) != 4:
                    continue
                ymin, xmin, ymax, xmax = roi
                ymin = int(ymin * wpercent)
                xmin = int(xmin * wpercent)
                ymax = int(ymax * wpercent)
                xmax = int(xmax * wpercent)
                resized_rois.append((ymin, xmin, ymax, xmax))
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
        if rois is not None:
            for roi in rois:
                ymin, xmin, ymax, xmax = roi
                ymin = int(ymin * hpercent)
                xmin = int(xmin * hpercent)
                ymax = int(ymax * hpercent)
                xmax = int(xmax * hpercent)
                resized_rois.append((ymin, xmin, ymax, xmax))
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=intermethod
    )  # this flips the desireable_size vector

    # Padding
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

    if rois is not None:
        return resized_img, resized_rois
    else:
        return resized_img


def get_imgs(img_path, scale, transform=None, multiscale=False, return_orig_img=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    orig_img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(orig_img)

    if return_orig_img:
        return img, orig_img
    else:
        return img


def get_tokenizer(llm_type):
    if llm_type == "gpt":
        tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        tokenizer.add_special_tokens(
            {
                "bos_token": "<|startoftext|>",
                "pad_token": "<|padtext|>",
                "mask_token": "<|masktext|>",
                "sep_token": "<|separatetext|>",
                "unk_token": "<|unknowntext|>",
                "additional_special_tokens": [
                    "<|keytext|>",
                ],
            }
        )
    elif llm_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "epfl-llm/meditron-7b", token=MY_API_TOKEN, padding_side="right"
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<KEY>",
                ],
            }
        )
    elif llm_type == "llama2":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", token=MY_API_TOKEN, padding_side="right"
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                "sep_token": "<sep>",
                "additional_special_tokens": [
                    "<KEY>",
                ],
            }
        )
    elif llm_type == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", token=MY_API_TOKEN
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": "<|pad_text|>",
                "mask_token": "<|mask_text|>",
                "sep_token": "<|separate_of_text|>",
                "additional_special_tokens": [
                    "<|keyword_of_text|>",
                ],
            }
        )
    else:
        tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        tokenizer.add_special_tokens(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "additional_special_tokens": [
                    "[KEY]",
                ],
            }
        )
    return tokenizer


def check_element_type(element, str_pool=None):
    if str_pool is None:
        # either non-empty string or non-nan float
        return (isinstance(element, str) and element != "") or (
            isinstance(element, float) and not np.isnan(element)
        )
    else:
        # either string in pool or non-nan float
        return (isinstance(element, str) and element in str_pool) or (
            isinstance(element, float) and not np.isnan(element)
        )


def get_specificity_with_sensitivity(y_true, y_prob, sensitivity):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(tpr - sensitivity))
    specificity = 1 - fpr[idx]
    return specificity


def pfbeta(labels, predictions, beta=1):
    """
    labels: (N,) binary
    predictions: (N,) class 1 score
    beta: int default=1, prob-F1 score
    """
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    # print(ctp, cfp)
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    # print(c_recall, c_precision)
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0
