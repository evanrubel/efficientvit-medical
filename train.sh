#!/bin/sh
#SBATCH -t 6:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=300GB
#SBATCH --partition=kellislab
#SBATCH --gres=gpu:a100:1

# S
# $DATA_PATH = "/om/user/sophiejg/project/tinyml_finalproj/CVPR24-MedSAMLaptopData/train_npz/"
# E
DATA_PATH="/data/rbg/users/erubel/efficient/efficientvit/data/CVPR24-MedSAMLaptopData/train_npz/"

torchrun --nproc_per_node=1 applications/efficientvit_sam/train_efficientvit_medsam_model.py applications/efficientvit_sam/configs/efficientvit_sam_xl1.yaml --data_provider.root $DATA_PATH --data_provider.dataset medsam --path .exp/efficientvit_sam/efficientvit_sam_xl1 --resume