#!/bin/sh
#SBATCH -t 2:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=200GB

BASE="/om/user/sophiejg/project/tinyml_finalproj"

python calc_acc.py --segs "$BASE/efficientvit-medical/exp/efficientvit_medsam/infer/efficientvit_medsam_l1/npz" --gts $BASE"/CVPR24-MedSAMLaptopData/validation-box/gts" --imgs "$BASE/CVPR24-MedSAMLaptopData/validation-box/imgs" --output_csv "$BASE/efficientvit-medical/exp/efficientvit_medsam/infer/efficientvit_medsam_l1/seg_metrics.csv"


