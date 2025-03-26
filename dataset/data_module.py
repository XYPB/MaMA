import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from .sampler import BalanceSampler


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        collate_fn,
        transforms,
        data_pct,
        batch_size,
        num_workers,
        img_size=518,
        crop_size=518,
        llm_type="gpt",
        ctx_prompt=False,
        cls_prompt=False,
        train_split="train",
        valid_split="valid",
        test_split="test",
        structural_cap=False,
        simple_cap=False,
        natural_cap=False,
        instance_test_cap=False,
        inter_view=False,
        inter_side=False,
        balanced_test=False,
        slip=False,
        balance_training=False,
        small_balanced_train=False,
        pred_density=False,
        load_jpg=False,
        mask_ratio=0.0,
        mask_meta=-1.0,
        balance_ratio=-1,
    ):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.crop_size = crop_size
        self.llm_type = llm_type
        self.ctx_prompt = ctx_prompt
        self.cls_prompt = cls_prompt
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.instance_test_cap = instance_test_cap
        self.inter_view = inter_view
        self.inter_side = inter_side
        self.balanced_test = balanced_test
        self.slip = slip
        self.balance_training = balance_training
        self.small_balanced_train = small_balanced_train
        self.pred_density = pred_density
        self.load_jpg = load_jpg
        self.mask_ratio = mask_ratio
        self.mask_meta = mask_meta
        self.balance_ratio = balance_ratio

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.img_size, self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            split=self.train_split,
            transform=transform,
            data_pct=self.data_pct,
            llm_type=self.llm_type,
            ctx_prompt=self.ctx_prompt,
            cls_prompt=self.cls_prompt,
            simple_cap=self.simple_cap,
            structural_cap=self.structural_cap,
            natural_cap=self.natural_cap,
            instance_test_cap=self.instance_test_cap,
            inter_side=self.inter_side,
            inter_view=self.inter_view,
            balanced_test=self.balanced_test,
            slip=self.slip,
            small_balanced_train=self.small_balanced_train,
            pred_density=self.pred_density,
            imsize=self.img_size,
            load_jpg=self.load_jpg,
            mask_ratio=self.mask_ratio,
            mask_meta=self.mask_meta,
        )

        if self.balance_training:
            if self.balance_ratio > 0:
                sampler = BalanceSampler(
                    np.array(dataset.labels), ratio=self.balance_ratio
                )
            else:
                num_samples = len(dataset)
                _, class_counts = np.unique(
                    list(dataset.path2label.values()), return_counts=True
                )
                class_weights = 1.0 / class_counts
                weights = []
                for idx in range(num_samples):
                    lb = dataset.path2label[dataset.filenames[idx]]
                    weights.append(class_weights[lb])

                sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.img_size, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split=self.valid_split,
            transform=transform,
            data_pct=self.data_pct,
            llm_type=self.llm_type,
            ctx_prompt=self.ctx_prompt,
            cls_prompt=self.cls_prompt,
            simple_cap=self.simple_cap,
            structural_cap=self.structural_cap,
            natural_cap=self.natural_cap,
            instance_test_cap=self.instance_test_cap,
            inter_side=self.inter_side,
            inter_view=self.inter_view,
            balanced_test=self.balanced_test,
            slip=self.slip,
            small_balanced_train=self.small_balanced_train,
            pred_density=self.pred_density,
            imsize=self.img_size,
            load_jpg=self.load_jpg,
        )
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.img_size, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split=self.test_split,
            transform=transform,
            data_pct=self.data_pct,
            llm_type=self.llm_type,
            ctx_prompt=self.ctx_prompt,
            cls_prompt=self.cls_prompt,
            simple_cap=self.simple_cap,
            structural_cap=self.structural_cap,
            natural_cap=self.natural_cap,
            instance_test_cap=self.instance_test_cap,
            inter_side=self.inter_side,
            inter_view=self.inter_view,
            balanced_test=self.balanced_test,
            slip=self.slip,
            small_balanced_train=self.small_balanced_train,
            pred_density=self.pred_density,
            imsize=self.img_size,
            load_jpg=self.load_jpg,
        )
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


# if __name__=="__main__":
