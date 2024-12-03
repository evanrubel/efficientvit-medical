import json
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools import mask as mask_utils
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from efficientvit.apps.data_provider import DataProvider
from efficientvit.samcore.data_provider.utils import (
    Normalize_and_Pad,
    RandomHFlip,
    ResizeLongestSide,
    SAMDistributedSampler,
)


import itertools
import os
import random
import zipfile
from glob import glob
from time import time
from typing import List, Optional

import albumentations as A

from efficientvit.samcore.data_provider.utils import (
    get_bbox,
    get_image_transform,
    resize_box,
    transform_gt,
    parmap
)


__all__ = ["MedSAMDataProvider"]


class MedSAMBaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_encoder_input_size: int = 512,
        prompt_encoder_input_size: Optional[int] = None,
        scale_image: bool = True,
        normalize_image: bool = False,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        interpolation: str = "bilinear",
    ):
        self.data_dir = data_dir
        self.image_encoder_input_size = image_encoder_input_size
        self.prompt_encoder_input_size = (
            prompt_encoder_input_size
            if prompt_encoder_input_size is not None
            else image_encoder_input_size
        )
        self.scale_image = scale_image
        self.normalize_image = normalize_image
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.interpolation = interpolation
        self.transform_image = get_image_transform(
            long_side_length=self.image_encoder_input_size,
            min_max_scale=self.scale_image,
            normalize=self.normalize_image,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            interpolation=self.interpolation,
        )


class MedSAMTrainDataset(MedSAMBaseDataset):
    def __init__(
        self,
        bbox_random_shift: int = 5,
        mask_num: int = 5,
        data_aug: bool = True,
        num_workers: int = 8,
        glob_pattern: str = "**/*.npz",
        limit_npz: Optional[int] = None,
        limit_sample: Optional[int] = None,
        aug_transform: Optional[A.TransformType] = None,
        train: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.train = train
        self.bbox_random_shift = bbox_random_shift
        self.mask_num = mask_num

        self.npz_file_paths = sorted(
            glob(os.path.join(self.data_dir, glob_pattern), recursive=True)
        )
        if limit_npz is not None:
            self.npz_file_paths = self.npz_file_paths[:limit_npz]

        self.items = list(
            itertools.chain.from_iterable(
                parmap(self.__flatten_npz, self.npz_file_paths, nprocs=num_workers)
            )
        )
        if limit_sample is not None:
            rng = random.Random(42)
            self.items = rng.sample(self.items, limit_sample)

        if self.train:
            self.items = self.items[: int(len(self.items) * 0.99)]
            print("Number of training samples:", len(self.items))
        else:
            self.items = self.items[int(len(self.items) * 0.99) :]
            print("Number of validation samples:", len(self.items))

        if not data_aug:
            self.aug_transform = A.NoOp()
        elif aug_transform is not None:
            self.aug_transform = aug_transform
        else:
            self.aug_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )

    def __flatten_npz(self, npz_file_path):
        try:
            data = np.load(npz_file_path, "r")
        except zipfile.BadZipFile:
            return []

        gts = data["gts"]
        assert len(gts.shape) == 2 or len(gts.shape) == 3
        if len(gts.shape) > 2:  # 3D
            return [
                (npz_file_path, slice_index)
                for slice_index in gts.max(axis=(1, 2)).nonzero()[0]
            ]
        else:  # 2D
            return [(npz_file_path, -1)] if gts.max() > 0 else []

    def get_name(self, item):
        name = os.path.basename(item[0]).split(".")[0]
        return name + f"_{item[1]:03d}" if item[1] != -1 else name

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        data = np.load(item[0], "r")
        img = data["imgs"]
        gt = data["gts"]  # multiple labels [0, 1, 4, 5, ...], (H, W)

        if item[1] != -1:  # 3D
            img = img[item[1], :, :]
            gt = gt[item[1], :, :]

        # duplicate channel if the image is grayscale
        if len(img.shape) < 3:
            img = np.repeat(img[..., None], 3, axis=-1)  # (H, W, 3)

        labels = np.unique(gt[gt > 0])
        assert len(labels) > 0, f"No label found in {item[0]}"
        # labels = random.choices(labels, k=self.mask_num)

        if self.train:
            # Randomly select `num_masks` from the available labels
            if len(labels) > self.mask_num:
                selected_labels = np.random.choice(labels, size=self.mask_num, replace=False)
            else:
                repeat, residue = self.mask_num // len(labels), self.mask_num % len(labels)
                selected_labels = np.concatenate([labels for _ in range(repeat)] + [np.random.choice(labels, size=residue, replace=False)])
        else:
            # Select first 'num_masks' for deterministic evaluation
            if len(labels) > self.mask_num:
                selected_labels = np.arange(self.mask_num)
            else:
                repeat, residue = self.mask_num // len(labels), self.mask_num % len(labels)
                selected_labels = np.concatenate([labels for _ in range(repeat)] + [np.arange(residue)])

        # augmentation
        all_masks = [np.array(gt == label, dtype=np.uint8) for label in selected_labels]
        augmented = self.aug_transform(image=img, masks=all_masks)
        img, all_masks = augmented["image"], augmented["masks"]
        original_size = img.shape[:2]

        # Extract boxes and masks from ground-truths
        masks_list = []
        boxes_list = []
        for mask in all_masks:
            mask = torch.from_numpy(mask).type(torch.uint8)
            mask = transform_gt(mask, self.image_encoder_input_size)
            if mask.max() == 0:
                H, W = mask.shape
                x_min = random.randint(0, W - 1)
                x_max = random.randint(0, W - 1)
                y_min = random.randint(0, H - 1)
                y_max = random.randint(0, H - 1)
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                if y_min > y_max:
                    y_min, y_max = y_max, y_min

                bbox_shift = 1
                x_min = max(0, x_min - bbox_shift)
                x_max = min(W - 1, x_max + bbox_shift)
                y_min = max(0, y_min - bbox_shift)
                y_max = min(H - 1, y_max + bbox_shift)

                box = np.array([x_min, y_min, x_max, y_max])
            else:
                box = get_bbox(mask, random.randint(0, self.bbox_random_shift))
            box = resize_box(box, mask.shape, self.prompt_encoder_input_size)
            box = torch.tensor(box, dtype=torch.float32)
            masks_list.append(mask)
            boxes_list.append(box)

        tsfm_img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.uint8)
        tsfm_img = self.transform_image(tsfm_img.unsqueeze(0)).squeeze(0)


        #TODO: add points
        return {
            "image": tsfm_img,  # (3, H, W)
            "masks": torch.stack(masks_list).unsqueeze(1),  # (N, H, W)
            "bboxs": torch.stack(boxes_list),  # (N, 4)
            "shape": torch.tensor(original_size, dtype=torch.int32),
        }



class MedSAMDataProvider(DataProvider):
    name = "medsam"

    def __init__(
        self,
        root: str,
        sub_epochs_per_epoch: int,
        num_masks: int,
        train_batch_size: int,
        test_batch_size: int,
        valid_size: Optional[int | float] = None,
        n_worker=8,
        image_size: int = 1024,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
    ):
        self.root = root
        self.num_masks = num_masks
        self.sub_epochs_per_epoch = sub_epochs_per_epoch

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )


    def build_aug_transform():
        """
        Constructs an augmentation pipeline using albumentations.
        """
        aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.2, 
                rotate_limit=90, 
                border_mode=0, 
                p=0.5
            ),
        ])
        return aug_transform

    def build_datasets(self) -> tuple[Any, Any, Any]:
        train_transform = self.build_train_transform()
        # valid_transform = self.build_valid_transform()

        # train_dataset = OnlineDataset(root=self.root, train=True, num_masks=self.num_masks, transform=train_transform)
        # val_dataset = OnlineDataset(root=self.root, train=False, num_masks=2, transform=valid_transform)

        # limit train sample to 1000 images while debugging
        limit_sample = 1000

        # Create training dataset
        train_dataset = MedSAMTrainDataset(
            data_dir=self.root,
            image_encoder_input_size=self.image_size[0],
            bbox_random_shift=5,
            mask_num=self.num_masks,
            limit_sample =limit_sample,
            data_aug=True,
            aug_transform=train_transform,
            glob_pattern="**/*.npz",  # Or customize based on your dataset structure
            num_workers=self.n_worker,
            train=True
        )

        # Create validation dataset (can use the same dataset class with different parameters)
        val_dataset = MedSAMTrainDataset(
            data_dir=self.root,
            image_encoder_input_size=self.image_size[0],
            bbox_random_shift=5,
            mask_num=2,  # Fewer masks for validation
            limit_sample=limit_sample,
            data_aug=False,  # No augmentation during validation
            # aug_transform=valid_transform,
            glob_pattern="**/*.npz",  # Same structure as training
            num_workers=self.n_worker,
            train=False
        )

        test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def build_dataloader(self, dataset: Optional[Any], batch_size: int, n_worker: int, drop_last: bool, train: bool):
        if dataset is None:
            return None
        if train:
            sampler = SAMDistributedSampler(dataset, sub_epochs_per_epoch=self.sub_epochs_per_epoch)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=True, num_workers=n_worker)
            return dataloader
        else:
            sampler = DistributedSampler(dataset, shuffle=False)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=False, num_workers=n_worker)
            return dataloader

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        if isinstance(self.train.sampler, SAMDistributedSampler):
            self.train.sampler.set_epoch_and_sub_epoch(epoch, sub_epoch)
