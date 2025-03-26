import torch
import time
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
from collections import Counter
from glob import glob
import torchvision.transforms as transforms
import random
import pydicom as dicom
from .transforms import OtsuCut
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from .constants_val import *
from .utils import get_imgs, read_from_dicom, get_tokenizer


class RSNAMammo(torch.utils.data.Dataset):

    def __init__(
        self,
        split="train",
        transform=None,
        data_pct=1.0,
        llm_type="gpt",
        max_words=72,
        img_size=1024,
        structural_cap=False,
        natural_cap=False,
        simple_cap=False,
        balanced_test=False,
        *args,
        **kwargs,
    ):
        if split == "test":
            split = "valid"
        assert split in ["train", "valid"]
        self.transform = transform
        self.llm_type = llm_type
        self.img_size = img_size
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.natural_cap = natural_cap
        self.simple_cap = simple_cap
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.n_classes = 2

        if split == "train":
            self.df = pd.read_csv(RSNA_MAMMO_TRAIN_CSV)
        else:
            if balanced_test:
                self.df = pd.read_csv(RSNA_MAMMO_BALANCE_TEST_CSV)
            else:
                self.df = pd.read_csv(RSNA_MAMMO_TEST_CSV)

        if data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=data_pct)
        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.path2label = {}
        self.labels = []
        missing_cnt = 0
        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            label = entry["cancer"]
            pid, iid = entry["patient_id"], entry["image_id"]
            path = os.path.join(RSNA_MAMMO_JPEG_DIR, f"{pid}/{iid}_resized.jpg")
            if not os.path.exists(path):
                missing_cnt += 1
            self.labels.append(label)
            self.filenames.append(path)
            self.path2label[path] = label
        print("### Sampled split distribution: ", Counter(self.labels))

    def __len__(self):
        return len(self.df)

    def get_caption(self, series_sents):
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        tokens["masked_ids"] = tokens["input_ids"]

        return tokens, x_len

    def get_zeroshot_caption(self):
        base_captions = ""
        zero_shot_caps = []
        zero_shot_caps_len = []
        for label, canncer_desc in RSNA_MAMMO_CANCER_DESC.items():
            # build density caption following training format
            birads, birads_desc = RSNA_MAMMO_BIRADS_DESC[label]
            if self.structural_cap:
                # findings
                captions = (
                    base_captions
                    + EMBED_FINDINGS
                    + EMBED_FINDS_CAPTION
                    + canncer_desc
                    + " "
                )
                # impression
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", birads
                ).replace("{{BIRADS_DESC}}", birads_desc)
                # overall assesment
                captions += EMBED_ASSESSMENT + birads_desc
            elif self.natural_cap:
                # findings
                captions = base_captions + EMBED_FINDS_CAPTION + canncer_desc + " "
                # impression
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", birads_desc)
            else:
                captions = (
                    base_captions
                    + BREAST_BASE_CAPTION
                    + BREAST_BIRADS_CAPTION
                    + str(birads)
                    + ": "
                    + birads_desc
                    + "."
                )
            # Update caption type if using raw style caption
            captions = captions.replace(".", self.tokenizer.sep_token)
            captions = captions.replace(";", self.tokenizer.sep_token)
            if self.llm_type != "bert":
                captions = (
                    self.tokenizer.bos_token + captions + self.tokenizer.eos_token
                )
            cap, cap_len = self.get_caption([captions])
            zero_shot_caps.append(cap)
            zero_shot_caps_len.append(cap_len)

        stacked_caps = {}
        for cap in zero_shot_caps:
            for k, v in cap.items():
                if k not in stacked_caps:
                    stacked_caps[k] = v
                else:
                    stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
        zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
        self.zero_shot_caps = stacked_caps
        self.zero_shot_caps_len = zero_shot_caps_len

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        label = self.labels[idx]
        path = self.filenames[idx]

        img = get_imgs(path, scale=self.img_size, transform=self.transform)
        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1
        if self.zero_shot_caps is None:
            self.get_zeroshot_caption()

        return (
            img,
            self.zero_shot_caps,
            self.zero_shot_caps_len,
            path,
            one_hot_labels,
            self.zero_shot_caps,
            one_hot_labels,
            img,
        )
