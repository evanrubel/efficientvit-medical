DATA_PATH="/data/rbg/users/erubel/efficient/efficientvit/data/CVPR24-MedSAMLaptopData/train_npz/"

torchrun --nproc_per_node=1 applications/efficientvit_sam/train_efficientvit_medsam_model.py applications/efficientvit_sam/configs/efficientvit_sam_xl1.yaml --data_provider.root $DATA_PATH --data_provider.dataset medsam --path .exp/efficientvit_sam/efficientvit_sam_xl1 --resume