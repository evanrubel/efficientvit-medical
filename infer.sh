#!/bin/sh
#SBATCH -t 2:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=300GB
#SBATCH --gres=gpu:a100:4

# S
BASE="/om/user/sophiejg/project/tinyml_finalproj"

DATA_PATH="$BASE/CVPR24-MedSAMLaptopData/validation-box/imgs"

MODEL="efficientvit-sam-l1"

WEIGHT_URL="$BASE/efficientvit-medical/assets/checkpoints/efficientvit_sam/distilled_model/efficientvit-sam-l1.pt"

OUTPUT_DIR="$BASE/efficientvit-medical/exp/efficientvit_medsam/infer/efficientvit_sam_l1"

torchrun --nproc_per_node=4 --master_port=29501  applications/efficientvit_sam/eval_efficientvit_medsam_model.py --model $MODEL --weight_url $WEIGHT_URL --image_size 512 --data_root $DATA_PATH --output_dir $OUTPUT_DIR --save_overlay True 
