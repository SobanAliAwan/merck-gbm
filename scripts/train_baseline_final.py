"""
Baseline 3D U-Net — Final Training Script
Best result: Mean Dice 0.8792 at epoch 46 (WT 0.9132, TC 0.8912, ET 0.8331)
Dataset: BraTS 2023 GLI (1251 cases, 80/10/10 split)
Hardware: Kaggle T4 GPU, ~22 min/epoch
Status: Training in progress (epoch 46/100)
"""

# Key findings during training:
# - BraTS 2023 uses labels 1,2,3 (not 1,2,4 as in older versions)
# - num_workers=0 required on Kaggle to prevent DataLoader hanging
# - DataParallel checkpoint keys have 'module.' prefix — strip when loading on single GPU
# - MONAI version mismatch: old checkpoint uses 'sub0/sub1', new uses 'submodule.0/submodule.1'
