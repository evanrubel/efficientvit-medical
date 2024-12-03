import math
import random
from copy import deepcopy
# from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


from typing import List, Optional, Tuple
import numpy as np
import torchvision.transforms.v2 as transforms


import multiprocessing
from tqdm import tqdm


__all__ = ["SAMDistributedSampler", "RandomHFlip", "ResizeLongestSide", "Normalize_and_Pad"]


class SAMDistributedSampler(DistributedSampler):
    """
    Modified from https://github.com/pytorch/pytorch/blob/97261be0a8f09bed9ab95d0cee82e75eebd249c3/torch/utils/data/distributed.py.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sub_epochs_per_epoch: int = 1,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.sub_epoch = 0
        self.sub_epochs_per_epoch = sub_epochs_per_epoch
        self.set_sub_num_samples()

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        indices = indices[(self.sub_epoch % self.sub_epochs_per_epoch) :: self.sub_epochs_per_epoch]

        return iter(indices)

    def __len__(self) -> int:
        return self.sub_num_samples

    def set_sub_num_samples(self) -> int:
        self.sub_num_samples = self.num_samples // self.sub_epochs_per_epoch
        if self.sub_num_samples % self.sub_epochs_per_epoch > self.sub_epoch:
            self.sub_num_samples += 1

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            sub_epoch (int): Sub epoch number.
        """
        self.epoch = epoch
        self.sub_epoch = sub_epoch
        self.set_sub_num_samples()


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            masks = torch.flip(masks, dims=[2])
            points = deepcopy(points).to(torch.float)
            bboxs = deepcopy(bboxs).to(torch.float)
            points[:, 0] = shape[-1] - points[:, 0]
            bboxs[:, 0] = shape[-1] - bboxs[:, 2] - bboxs[:, 0]

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}


class ResizeLongestSide(object):
    """
    Modified from https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/transforms.py.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:
        target_size = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        return F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)

    def apply_boxes(self, boxes: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords(self, coords: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        image = self.apply_image(image.unsqueeze(0), shape).squeeze(0)
        masks = self.apply_image(masks.unsqueeze(1), shape).squeeze(1)
        points = self.apply_coords(points, shape)
        bboxs = self.apply_boxes(bboxs, shape)

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}


class Normalize_and_Pad(object):
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
        self.transform = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        h, w = image.shape[-2:]
        image = self.transform(image)

        padh = self.target_length - h
        padw = self.target_length - w

        image = F.pad(image.unsqueeze(0), (0, padw, 0, padh), value=0).squeeze(0)
        masks = F.pad(masks.unsqueeze(1), (0, padw, 0, padh), value=0).squeeze(1)

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}
    



class ResizeLongestSide_Med(torch.nn.Module):
    def __init__(
        self,
        long_side_length: int,
        interpolation: str,
    ) -> None:
        super().__init__()
        self.long_side_length = long_side_length
        self.interpolation = interpolation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        oldh, oldw = image.shape[-2:]
        if max(oldh, oldw) == self.long_side_length:
            return image
        newh, neww = self.get_preprocess_shape(oldh, oldw, self.long_side_length)
        return F.interpolate(
            image, (newh, neww), mode=self.interpolation, align_corners=False
        )

    @staticmethod
    def get_preprocess_shape(
        oldh: int,
        oldw: int,
        long_side_length: int,
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class MinMaxScale(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image should have shape (..., 3, H, W)
        """
        assert len(image.shape) >= 3 and image.shape[-3] == 3
        min_val = image.amin((-3, -2, -1), keepdim=True)
        max_val = image.amax((-3, -2, -1), keepdim=True)
        return (image - min_val) / torch.clip(max_val - min_val, min=1e-8, max=None)


class PadToSquare(torch.nn.Module):
    def __init__(self, target_size: int) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        return F.pad(image, (0, self.target_size - w, 0, self.target_size - h), value=0)


def get_bbox(mask: np.ndarray, bbox_shift: int = 0) -> np.ndarray:
    """
    Get the bounding box coordinates from the mask

    Parameters
    ----------
    mask : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W - 1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H - 1, y_max + bbox_shift)

    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


def resize_box(
    box: np.ndarray,
    original_size: Tuple[int, int],
    prompt_encoder_input_size: int,
) -> np.ndarray:
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image
    prompt_encoder_input_size : int
        the target size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = prompt_encoder_input_size / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


def get_image_transform(
    long_side_length: int,
    min_max_scale: bool = True,
    normalize: bool = False,
    pixel_mean: Optional[List[float]] = None,
    pixel_std: Optional[List[float]] = None,
    interpolation: str = "bilinear",
) -> transforms.Transform:
    tsfm = [
        ResizeLongestSide_Med(long_side_length, interpolation),
        transforms.ToDtype(dtype=torch.float32, scale=False),
    ]
    if min_max_scale:
        tsfm.append(MinMaxScale())
    if normalize:
        tsfm.append(transforms.Normalize(pixel_mean, pixel_std))
    tsfm.append(PadToSquare(long_side_length))
    return transforms.Compose(tsfm)


def transform_gt(gt: torch.Tensor, long_side_length: int):
    gt = gt[None, None, ...]
    oldh, oldw = gt.shape[-2:]
    newh, neww = ResizeLongestSide_Med.get_preprocess_shape(oldh, oldw, long_side_length)
    gt = F.interpolate(gt, (newh, neww), mode="nearest-exact")
    gt = F.pad(gt, (0, long_side_length - neww, 0, long_side_length - newh), value=0)
    return gt.squeeze((0, 1))


# from src.utils.multiprocessing in medficientsam
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=fun, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [
        q_in.put((i, x)) for i, x in enumerate(tqdm(X, position=0, desc="Queue In"))
    ]
    for _ in range(nprocs):
        q_in.put((None, None))
    res = [q_out.get() for _ in tqdm(range(len(sent)), position=1, desc="Queue Out")]
    res = [x for _, x in sorted(res)]
    for p in proc:
        p.join()

    return res

