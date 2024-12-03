#!/bin/sh
#SBATCH -t 6:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=300GB
#SBATCH --partition=kellislab
#SBATCH --gres=gpu:a100:2

torchrun --nproc_per_node=8 applications/efficientvit_sam/train_efficientvit_medsam_model.py applications/efficientvit_sam/configs/efficientvit_sam_xl1.yaml --data_provider.root /om/user/sophiejg/project/tinyml_finalproj/CVPR24-MedSAMLaptopData/train_npz/ --data_provider.dataset medsam --path .exp/efficientvit_sam/efficientvit_sam_xl1 --resume
