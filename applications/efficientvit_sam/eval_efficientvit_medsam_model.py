import argparse
import json
import os
import sys

import pdb 

import logging
import numpy as np
import torch
from lvis import LVIS
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from sam_eval_utils import Clicker, evaluate_predictions_on_coco, evaluate_predictions_on_lvis, get_iou_metric, iou
from visualize_utils import visualize_output
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from pathlib import Path
from time import perf_counter_ns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.samcore.data_provider import MedSAMTestDataset





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


def bbox_xywh_to_xyxy(bbox: list[int]) -> list[int]:
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def ann_to_mask(ann, h, w):
    if type(ann["segmentation"]) == list:
        rles = mask_util.frPyObjects(ann["segmentation"], h, w)
        rle = mask_util.merge(rles)
    elif type(ann["segmentation"]["counts"]) == list:
        rle = mask_util.frPyObjects(ann["segmentation"], h, w)
    else:
        raise NotImplementedError()

    mask = mask_util.decode(rle) > 0

    return mask


def sync_output(world_size, output):
    outs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(outs, output)
    merged_outs = []
    for sublist in outs:
        merged_outs += sublist

    return merged_outs


def predict_mask_from_box(predictor: EfficientViTSamPredictor, bbox: np.ndarray) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


def predict_mask_from_point(
    predictor: EfficientViTSamPredictor, point_coords: np.ndarray, point_labels: np.ndarray
) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


# class eval_dataset(Dataset):
#     def __init__(self, dataset, image_root, prompt_type, annotation_json_file, source_json_file=None):
#         self.dataset = dataset
#         self.image_root = image_root
#         self.prompt_type = prompt_type
#         self.annotation_json_file = annotation_json_file

#         if self.dataset == "coco":
#             self.images = os.listdir(self.image_root)
#             self.images = [os.path.join(self.image_root, image) for image in self.images]
#             self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]
#         elif self.dataset == "lvis":
#             self.images = json.load(open(self.annotation_json_file, "r"))["images"]
#             self.images = [
#                 os.path.join(self.image_root, image["coco_url"].split("/")[-2], image["coco_url"].split("/")[-1])
#                 for image in self.images
#             ]
#             self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]

#         elif self.dataset == "cvpr":

#         else:
#             raise NotImplementedError()

#         if self.prompt_type == "point" or self.prompt_type == "box":
#             self.annotations = json.load(open(self.annotation_json_file, "r"))["annotations"]
#         elif self.prompt_type == "box_from_detector":
#             self.source_json_file = json.load(open(source_json_file))
#         else:
#             raise NotImplementedError()

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_path = self.images[idx]
#         if self.prompt_type == "point" or self.prompt_type == "box":
#             anns = [ann for ann in self.annotations if ann["image_id"] == self.ids[idx]]
#             return {"image_path": image_path, "anns": anns}
#         elif self.prompt_type == "box_from_detector":
#             detections = [det for det in self.source_json_file if det["image_id"] == self.ids[idx]]
#             return {"image_path": image_path, "detections": detections}
#         else:
#             raise NotImplementedError()


def collate_fn(batch):
    return batch


def infer_2D(predictor, data, device, output_dir: Path, save_overlay: bool):
    npz_name = data['npz_name']
    img = data["image"] # (H, W, 3)
    boxes = data["boxes"]
    # original_size = data["original_size"].tolist()
    # new_size = data["new_size"].tolist()


    original_size = img.shape[:2]
    predictor.set_image(img)
    segs = np.zeros(original_size, dtype=np.uint16)
    start = perf_counter_ns()
    for idx, box in enumerate(boxes, start=1):
        pre_mask = predict_mask_from_box(predictor, box)
        # print(f"box shape: {box} ")
        if(np.max(pre_mask)>0):
            print(f"box shape: {box} ")
            print("predicted mask")
            print(pre_mask.shape)
        # mask, _ = model.prompt_and_decoder(image_embedding, box.unsqueeze(0))
        # reshape to original size for saving in output
        # mask = model.postprocess_masks(mask, new_size, original_size)
        # mask = mask.squeeze((0, 1)).cpu().numpy()
        segs[pre_mask > 0] = idx
    end = perf_counter_ns()

    np.savez_compressed(output_dir / "npz" / npz_name, segs=segs)

    if save_overlay:
        visualize_output(
            img=data["image"],
            boxes=data["boxes"],
            segs=segs,
            save_file=(output_dir / "png" / npz_name).with_suffix(".png"),
        )

    elapsed_time = (end - start) / 1000000 
    return elapsed_time
    


def infer_3D(predictor, data, device, output_dir: Path, save_overlay: bool):
    npz_name = data['npz_name']
    img_3D = data["image"] # (D, H, W, 3)
    print(img_3D.shape)
    boxes = data["boxes"] # (N, 6), [[x_min, y_min, z_min, x_max, y_max, z_max]]
    # original_size = data["original_size"].tolist()
    # new_size = data["new_size"].tolist()


    original_size = img_3D.shape[1:3]
    segs = np.zeros((img_3D.shape[0], *original_size), dtype=np.uint16)


    total_elapsed_time = 0
    for idx, box3D in enumerate(boxes, start=1):
        segs_i = np.zeros_like(segs, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        box_default = np.array([x_min, y_min, x_max, y_max])
        z_middle = (z_max + z_min) // 2

        # infer from middle slice to the z_max
        box_2D = box_default
        for z in range(int(z_middle), int(z_max)):
            img_2d = img_3D[z, :, :, :].squeeze()  # (H, W, 3)
            # print(img_2d.shape)
            predictor.set_image(img_2d)
            start = perf_counter_ns()
            pre_mask = predict_mask_from_box(predictor, box_2D)
            end = perf_counter_ns()
            elapsed_time = (end - start) / 1000000 
            total_elapsed_time += elapsed_time
            if np.max(pre_mask) > 0:
                box_2D = get_bbox(pre_mask)
                # box_2D = resize_box(
                #     box=box_2D,
                #     original_size=original_size,
                #     prompt_encoder_input_size=prompt_encoder_input_size,
                # )
                segs_i[z, pre_mask > 0] = 1
            else:
                box_2D = box_default

        # infer from middle slice to the z_min
        if np.max(segs_i[int(z_middle), :, :]) == 0:
            box_2D = box_default
        else:
            box_2D = get_bbox(segs_i[int(z_middle), :, :])
            # box_2D = resize_box(
            #     box=box_2D,
            #     original_size=original_size,
            #     prompt_encoder_input_size=prompt_encoder_input_size,
            # )

        for z in range(int(z_middle - 1), int(z_min - 1), -1):
            img_2d = img_3D[z, :, :, :].squeeze()  # (H, W, 3)
            predictor.set_image(img_2d)
            start = perf_counter_ns()
            pre_mask = predict_mask_from_box(predictor, box_2D)
            end = perf_counter_ns()
            elapsed_time = (end - start) / 1000000 
            total_elapsed_time += elapsed_time

            if np.max(pre_mask) > 0:
                box_2D = get_bbox(pre_mask)
                # box_2D = resize_box(
                #     box=box_2D,
                #     original_size=original_size,
                #     prompt_encoder_input_size=prompt_encoder_input_size,
                # )
                segs_i[z, pre_mask > 0] = 1
            else:
                box_2D = box_default

        segs[segs_i > 0] = idx

    np.savez_compressed(output_dir / "npz" / npz_name, segs=segs)

    # visualize image, mask and bounding box
    if save_overlay:
        z = segs.shape[0] // 2
        visualize_output(
            img=data["image"][z],
            boxes=data["boxes"][:, [0, 1, 3, 4]],
            segs=segs[z],
            save_file=(output_dir / "png" / npz_name).with_suffix(".png"),
        )

    return total_elapsed_time


# def run_box(efficientvit_sam, dataloader, local_rank):
#     efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
#     predictor = EfficientViTSamPredictor(efficientvit_sam)

#     output = []
#     for _, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
#         data = data[0]

#         # load image file in rgb format
#         sam_image = data['image'].cuda()
#         masks = data["masks"].cuda()

#         # look into why we need to * 2
#         bboxs = data["bboxs"].cuda() * 2 if sam_image.shape[2] == 512 else data["bboxs"].cuda()

#         predictor.set_image(sam_image)

#         pre_mask = predict_mask_from_box(predictor, bboxs)
#         miou = iou(pre_mask, masks)


#         # TODO: look into why they format is for ann in anns. In our case, is the case
#         # of multiple masks per image already taken care of in the data loader?
#         # should we treat each mask as a separate one?? yes, since they just append it all to a list
#         # one for each bbox in this method

#         # or should we al
#         # anns = data["anns"]
#         # # for every mask
#         # for ann in anns:
#         #     if ann["area"] < 1:
#         #         continue

#         #     # get sam mask
#         #     sam_mask = ann_to_mask(ann, sam_image.shape[0], sam_image.shape[1])

#         #     # get bbox
#         #     bbox = np.array(bbox_xywh_to_xyxy(ann["bbox"]))
#         #     #predict maask
#         #     pre_mask = predict_mask_from_box(predictor, bbox)

#         #     # evaluate predicted mask vs gt mask 
#         #     miou = iou(pre_mask, sam_mask)

#         #     result = {
#         #         "area": ann["area"],
#         #         "iou": miou,
#         #     }

#         #     output.append(result)

#     world_size = int(os.environ["WORLD_SIZE"])
#     merged_outs = sync_output(world_size, output)

#     return merged_outs





# def evaluate(results, prompt_type, dataset, annotation_json_file=None):
#     if prompt_type == "point" or prompt_type == "box":
#         print(", ".join([f"{key}={val:.3f}" for key, val in get_iou_metric(results).items()]))
#     elif prompt_type == "box_from_detector":
#         iou_type = "segm"
#         if dataset == "coco":
#             coco_api = COCO(annotation_json_file)
#             evaluate_predictions_on_coco(coco_gt=coco_api, coco_results=results, iou_type=iou_type)
#         elif dataset == "lvis":
#             lvis_api = LVIS(annotation_json_file)
#             evaluate_predictions_on_lvis(lvis_gt=lvis_api, lvis_results=results, iou_type=iou_type)
#     else:
#         raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--output_dir", type=str)
    # parser.add_argument("--annotation_json_file", type=str)
    # parser.add_argument("--source_json_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_overlay", type=bool, default=False)
    args = parser.parse_args()

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # setup.setup_dist_env()
    # setup.setup_seed(args.manual_seed, args.resume)
   
    # config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    
    efficientvit_sam = create_efficientvit_sam_model(args.model, True, args.weight_url, image_size = args.image_size)
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
    predictor = EfficientViTSamPredictor(efficientvit_sam)

    test_dataset = MedSAMTestDataset(
                data_dir=args.data_root,
                glob_pattern="**/*.npz",
                limit_npz=30
                )
    
    sampler = DistributedSampler(test_dataset, shuffle=False)
    dataloader = DataLoader(
        test_dataset, batch_size=1, sampler=sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn
    )

    output_dir = Path(args.output_dir)
    (output_dir / "npz").mkdir(parents=True, exist_ok=True)

    if args.save_overlay:
        (output_dir / "png").mkdir(parents=True, exist_ok=True)

    # Set up logging to a .log file
    logging.basicConfig(
        filename=os.path.join(output_dir,"prediction_times.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    prediction_times = {"2D": [], "3D": []}

    for _, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
        data = data[0]
        if data["image_type"] == "2D":
            elapsed_time = infer_2D(predictor, data, local_rank, output_dir, args.save_overlay)
            
        elif data["image_type"] == "3D":
            elapsed_time = infer_3D(predictor, data, local_rank, output_dir, args.save_overlay)
            
        else:
            raise NotImplementedError("Only support 2D and 3D image")
        
        prediction_times[data["image_type"]].append(elapsed_time)
        logging.info(f"Predicted {data['npz_name']} ({data['image_type']}) in {elapsed_time:.2f}s")
    

    # Compute and log average prediction times
    avg_times = {k: np.mean(v) if v else 0 for k, v in prediction_times.items()}
    logging.info(f"Average Prediction Time for 2D Images: {avg_times['2D']:.2f}s")
    logging.info(f"Average Prediction Time for 3D Images: {avg_times['3D']:.2f}s")

    print(f"Average Prediction Time for 2D Images: {avg_times['2D']:.2f}s")
    print(f"Average Prediction Time for 3D Images: {avg_times['3D']:.2f}s")


    torch.distributed.destroy_process_group()
    

    # build dataset using
    # eval_dataset = MedSAMTrainDataset(
    #     data_dir=args.data_root,
    #     image_encoder_input_size=args.image_size[0],
    #     bbox_random_shift=0,
    #     mask_num=args.num_masks,
    #     data_aug=False,
    #     glob_pattern="**/*.npz",  # Or customize based on your dataset structure
    #     num_workers=args.num_workers,
    #     train=False,
    #     test=True
    # )

    # # return {
    # #         "image": tsfm_img,  # (3, H, W)
    # #         "masks": torch.stack(masks_list).unsqueeze(1),  # (N, H, W)
    # #         "bboxs": torch.stack(boxes_list),  # (N, 4)
    # #         "shape": torch.tensor(original_size, dtype=torch.int32),
    # #     }

    # # dataset = eval_dataset(
    # #     args.dataset, args.image_root, args.prompt_type, args.annotation_json_file, args.source_json_file
    # # )
    # sampler = DistributedSampler(eval_dataset, shuffle=False)
    # dataloader = DataLoader(
    #     eval_dataset, batch_size=1, sampler=sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn
    # )

  
    # results = run_box(efficientvit_sam, dataloader, local_rank)
 

    # replace with cal_acc
    # if local_rank == 0:
    #     evaluate(results, args.prompt_type, args.dataset, args.annotation_json_file)
