import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_generalized_dice, compute_surface_dice
from tqdm import tqdm


def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max() + 1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(
            float(
                compute_generalized_dice(
                    torch.tensor(seg_i).unsqueeze(0).unsqueeze(0),
                    torch.tensor(gt_i).unsqueeze(0).unsqueeze(0),
                )[0]
            )
        )
    return np.mean(dsc)


def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max() + 1):
        gt_i = torch.tensor(gt == i)
        seg_i = torch.tensor(seg == i)
        nsd.append(
            float(
                compute_surface_dice(
                    seg_i.unsqueeze(0).unsqueeze(0),
                    gt_i.unsqueeze(0).unsqueeze(0),
                    class_thresholds=[tolerance],
                    spacing=spacing,
                )
            )
        )
    return np.mean(nsd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segs",
        required=True,
        type=str,
        help="directory of segmentation output",
    )
    parser.add_argument(
        "--gts",
        required=True,
        type=str,
        help="directory of ground truth",
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="directory of original images",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="seg_metrics.csv",
        help="output CSV file",
    )
    args = parser.parse_args()

    seg_metrics = OrderedDict()
    seg_metrics["case"] = []
    seg_metrics["model_type"] = []  # New column to distinguish 2D and 3D models
    seg_metrics["dsc"] = []
    seg_metrics["nsd"] = []

    segs = sorted(Path(args.segs).glob("*.npz"))

    for seg_file in tqdm(segs):
        gt_file = Path(args.gts) / seg_file.name
        img_file = Path(args.imgs) / seg_file.name
        if not gt_file.exists() or not img_file.exists():
            continue

        npz_seg = np.load(seg_file, "r")
        npz_gt = np.load(gt_file, "r")

        seg = npz_seg["segs"]
        gt = npz_gt["gts"] if "gts" in npz_gt else npz_gt["segs"]
        dsc = compute_multi_class_dsc(gt, seg)

        if seg_file.name.startswith("3D"):
            model_type = "3D"
            npz_img = np.load(img_file, "r")
            spacing = npz_img["spacing"]
            nsd = compute_multi_class_nsd(gt, seg, spacing)
        else:
            model_type = "2D"
            spacing = [1.0, 1.0, 1.0]
            nsd = compute_multi_class_nsd(
                np.expand_dims(gt, -1), np.expand_dims(seg, -1), spacing
            )

        seg_metrics["case"].append(seg_file.name)
        seg_metrics["model_type"].append(model_type)
        seg_metrics["dsc"].append(np.round(dsc, 4))
        seg_metrics["nsd"].append(np.round(nsd, 4))

    # Compute overall averages for 2D and 3D
    df = pd.DataFrame(seg_metrics)

    for model_type in ["2D", "3D"]:
        df_subset = df[df["model_type"] == model_type]
        if not df_subset.empty:
            avg_dsc = df_subset["dsc"].mean()
            avg_nsd = df_subset["nsd"].mean()
            seg_metrics["case"].append(f"{model_type}_average")
            seg_metrics["model_type"].append(model_type)
            seg_metrics["dsc"].append(avg_dsc)
            seg_metrics["nsd"].append(avg_nsd)

    # Write to CSV
    df = pd.DataFrame(seg_metrics)
    df.to_csv(args.output_csv, index=False, na_rep="NaN")

    print(f"Metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()
