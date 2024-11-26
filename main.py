import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image
import random
import SimpleITK as sitk
import shutil
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

### Dataset Generation ###

# Utility to normalize the image
def normalize_image(image):
    """Normalize CT scan pixel values to [0, 1]."""
    image = np.clip(image, -100, 250)  # Clipping Hounsfield Units
    image = (image - (-100)) / (250 - (-100))  # Scale to [0, 1]
    return image

# Preprocessing function
def preprocess_and_save(image_path, label_path, output_folder, num_slices=40, duplicate_channels=False):
    """
    Preprocess a single 3D image and label pair.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.
        output_folder (str): Path to save processed slices.
        num_slices (int): Number of random slices to sample per 3D image.
        duplicate_channels (bool): If True, duplicate the slice to create 3 channels.
    """
    # Load image and label
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Convert to numpy arrays
    image_np = sitk.GetArrayFromImage(image)  # Shape: (Depth, Height, Width)
    label_np = sitk.GetArrayFromImage(label)

    # Normalize image
    image_np = normalize_image(image_np)

    # Get random slice indices
    total_slices = image_np.shape[0]
    slice_indices = random.sample(range(total_slices), min(num_slices, total_slices))

    for i in slice_indices:
        img_slice = image_np[i, :, :]  # Take full 512x512 slice
        lbl_slice = label_np[i, :, :]  # Take corresponding label slice

        if duplicate_channels:
            img_slice = np.stack([img_slice] * 3, axis=-1)  ################## Duplicate to create 3 channels ##################

        # Save slices
        slice_name = f"slice_{os.path.basename(image_path).split('.')[0]}_{i:03d}"
        img_save_path = os.path.join(output_folder, "images", f"{slice_name}.png")
        lbl_save_path = os.path.join(output_folder, "labels", f"{slice_name}.png")

        # Convert to PIL Image and save
        Image.fromarray((img_slice * 255).astype(np.uint8)).save(img_save_path)
        Image.fromarray(lbl_slice.astype(np.uint8)).save(lbl_save_path)


def generate_dataset(args):
    for train_or_test in ['Tr']:
        input_folder = f"{args.input_path}/images{train_or_test}"
        label_folder = f"{args.input_path}/labels{train_or_test}"
        output_folder = f"{args.input_path}/processed_data{train_or_test}"  # Path to save processed data

        # Remove the folder if it exists
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Recreate the empty folders
        os.makedirs(os.path.join(output_folder, "images"))
        os.makedirs(os.path.join(output_folder, "labels"))

        # Process all files
        image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.nii.gz')])
        label_files = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.nii.gz')])

        for img_file, lbl_file in tqdm(zip(image_files, label_files), total=len(image_files)):
            # Potentially set duplicate_channels to False if you want
            preprocess_and_save(img_file, lbl_file, output_folder, num_slices=40, duplicate_channels=True)

        print("Preprocessing complete. Processed data saved to:", output_folder)


class InputDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        # self.transform = transform or ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        lbl_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load image and label
        image = torch.tensor(load_image(img_path))#.convert("L")  # Grayscale
        label = torch.tensor(load_image(lbl_path))#.convert("L")

        # image = torch.tensor(load_image("assets/fig/cat.jpg"))
        # image = np.array(Image.open("assets/fig/cat.jpg").convert("RGB"))

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(lbl_path).convert("RGB"))

        print(image.shape)
        print(label.shape)


        # Apply transforms
        # if self.transform:
        #     image = self.transform(image)
        #     label = self.transform(label)

        return image, label
    
    def get_image_name(self, idx):
        """Returns the image name without the file format extension."""

        return self.image_files[idx].split(".")[0]

def process_images(args, opt):
    """
    Process images using EfficientViT SAM
    """
    # Create output directory if it doesn't exist
    print(args.output_directory)
    os.makedirs(args.output_directory, exist_ok=True)

    # Build SAM model
    efficientvit_sam = create_efficientvit_sam_model(
        args.model,
        True, 
        args.weight_url
    ).to(f"cuda:{str(args.device)}").eval()
    
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
        efficientvit_sam,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator),
    )

    # Load dataset
    image_dir = f"{args.processed_data_path}/images"
    label_dir = f"{args.processed_data_path}/labels"

    # Create dataset instance
    dataset = InputDataset(image_dir, label_dir)
    
    # Process images and generate masks
    # for idx in range(len(dataset)):
    for idx in range(1):
        raw_image, label = dataset[idx]
        H, W, _ = raw_image.shape
        print(f"Processing image {idx+1}/{len(dataset)}: Size W={W}, H={H}")

        # Timestamp for unique temporary file
        tmp_file = f".tmp_{time.time()}.png"

        if args.mode == "all":
            masks = efficientvit_mask_generator.generate(raw_image)
            plt.figure(figsize=(20, 20))
            plt.imshow(raw_image)
            show_anns(masks)
            plt.axis("off")
            plt.savefig(os.path.join(args.output_directory, f"{dataset.get_image_name(idx)}.png"), format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
        # elif args.mode == "point":
        #     args.point = yaml.safe_load(f"[[{W // 2},{H // 2},{1}]]" if args.point is None else args.point)
        #     point_coords = [(x, y) for x, y, _ in args.point]
        #     point_labels = [l for _, _, l in args.point]

        #     efficientvit_sam_predictor.set_image(raw_image)
        #     masks, _, _ = efficientvit_sam_predictor.predict(
        #         point_coords=np.array(point_coords),
        #         point_labels=np.array(point_labels),
        #         multimask_output=args.multimask,
        #     )
        #     plots = [
        #         draw_scatter(
        #             draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
        #             point_coords,
        #             color=["g" if l == 1 else "r" for l in point_labels],
        #             s=10,
        #             ew=0.25,
        #             tmp_name=tmp_file,
        #         )
        #         for binary_mask in masks
        #     ]
        #     plots = cat_images(plots, axis=1)
        #     Image.fromarray(plots).save(args.output_path)
        # elif args.mode == "box":
        #     args.box = yaml.safe_load(args.box)
        #     efficientvit_sam_predictor.set_image(raw_image)
        #     masks, _, _ = efficientvit_sam_predictor.predict(
        #         point_coords=None,
        #         point_labels=None,
        #         box=np.array(args.box),
        #         multimask_output=args.multimask,
        #     )
        #     plots = [
        #         draw_bbox(
        #             draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
        #             [args.box],
        #             color="g",
        #             tmp_name=tmp_file,
        #         )
        #         for binary_mask in masks
        #     ]
        #     plots = cat_images(plots, axis=1)
        #     Image.fromarray(plots).save(args.output_path)
        else:
            raise NotImplementedError

def main():
    SPLEEN_DIR = "/data/rbg/users/erubel/efficient/efficientvit/data/Task09_Spleen"

    parser = argparse.ArgumentParser(description="EfficientViT SAM Medical Image Segmentation")
    parser.add_argument("--model", type=str, default="efficientvit-sam-xl1", help="SAM model variant")
    parser.add_argument("--weight_url", type=str, default=None, help="Custom weight URL")
    parser.add_argument("--multimask", action="store_true", help="Generate multiple masks")
    parser.add_argument("--input_path", type=str, default=SPLEEN_DIR, help="Directory of input data")
    parser.add_argument("--processed_data_path", type=str, default=os.path.join(SPLEEN_DIR, "processed_dataTr"), help="Directory of processed input data")
    parser.add_argument("--output_directory", type=str, default=".demo/medical_image_segmentation", help="Output directory")
    
    # Extended mode options to match original script
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["auto", "point", "box", "all"], 
                        help="Segmentation mode")
    
    # Optional box specification
    parser.add_argument("--box", type=str, default=None, 
                        help="Custom bounding box in YAML format, e.g. '[[x1,y1,x2,y2]]'")
    
    # Mask generation hyperparameters
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--min_mask_region_area", type=float, default=100)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--generate_dataset", type=bool, default=False)

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    if args.generate_dataset:
        generate_dataset(args)
    process_images(args, opt)

# Utility functions
def show_anns(anns) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas

def cat_images(image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
    shape_list = [image.shape for image in image_list]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(image_list):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
        image_list[i] = canvas

    image = np.concatenate(image_list, axis=axis)
    return image

def draw_bbox(
    image: np.ndarray,
    bbox: list[list[int]],
    color: str | list[str] = "g",
    linewidth=1,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in bbox]
    for (x0, y0, x1, y1), c in zip(bbox, color):
        plt.gca().add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=linewidth, edgecolor=c, facecolor=(0, 0, 0, 0)))
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image

def draw_scatter(
    image: np.ndarray,
    points: list[list[int]],
    color: str | list[str] = "g",
    marker="*",
    s=10,
    ew=0.25,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in points]
    for (x, y), c in zip(points, color):
        plt.scatter(x, y, color=c, marker=marker, s=s, edgecolors="white", linewidths=ew)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image

def load_image(data_path: str, mode="rgb") -> np.ndarray:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return np.array(img)

if __name__ == "__main__":
    main()