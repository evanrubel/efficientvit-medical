#!/bin/sh
#SBATCH -t 2:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=300GB
#SBATCH --gres=gpu:a100:4

# S
 DATA_PATH="/om/user/sophiejg/project/tinyml_finalproj/CVPR24-MedSAMLaptopData/train_npz/"
# E
#DATA_PATH="/data/rbg/users/erubel/efficient/efficientvit/data/CVPR24-MedSAMLaptopData/train_npz/"

torchrun --nproc_per_node=4 applications/efficientvit_sam/train_efficientvit_medsam_model.py applications/efficientvit_sam/configs/efficientvit_sam_l1.yaml --data_provider.root $DATA_PATH --data_provider.dataset medsam --path exp/efficientvit_medsam/efficientvit_medsam_l1_test --resume
