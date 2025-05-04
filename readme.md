## Introduction to SSDI
This repository provides the code and resources for the Image Grammar project, which implements the methods described in the [paper](https://arxiv.org/abs/2401.17515) titled *"Semantic-Syntactic Discrepancy in Images (SSDI): Learning Meaning and Order of Features from Natural Images"*. It includes tools to reproduce the results presented in the paper, focusing on the proposed two-stage semantic and syntactic learning method and SSDI detection. 

The project addresses discrepancies in images within specific object classes (e.g., faces in CelebA and CelebAHQ) and scene classes (e.g., rooms in SUN-RGBD and SUN-RGBD-13-Classes). 

With this repository, you can:
- Generate SSDI attacks for images in the CelebA and SUN-RGBD datasets
- Train and run inference using the two-stage PiCIE and bi-LSTM models for SSDI learning
- Reproduce SSDI detection results for the CelebA and SUN-RGBD datasets

## Setup
Set-ups for this project involve preparing datasets and downloading the pretrained model weights.    

### Prepare CelebA dataset
Download the CelebAMask-HQ Dataset from [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) and decompress.
Prepare CelebAHQ images and labels in your workspace path following the below format:
```python
img_path = "YOUR_PATH/CelebAMask-HQ/CelebA-HQ-img"  
label_path = "YOUR_PATH/CelebAMask-HQ/CelebA-HQ-label-coarse"  
```
Then, in `CelebA_syntax/configs/config_test_LSTM.py`, `class config_test`, change `dataset_dir = YOUR_PATH`.

### Prepare CelebA model weights:  
- Download Resnet18+FPN Segmentation model from [Resnet18+FPN GDrive Link](https://drive.google.com/uc?export=download&id=1GkmjmegVvsdDMwO5C0uPfoOHueg0ts6w), and put the file under `CelebA_syntax/models_trained_on_cropped_masks`.  
Note: The model path is defined and loaded as `--eval_path models_trained_on_cropeed_masks/celeba_resnet50_checkpoint_20.pth.tar` in `CelebA_syntax/scripts/test_lstm_for_paper.sh`.  
- Download pretrained Bi-LSTM model from [CelebA Bi-LSTM GDrive Link](https://drive.google.com/uc?export=download&id=149X1gdal5y7mRtp1oC8CAASc1qFybagv), and put the file under `CelebA_syntax/models_trained_on_cropped_masks`.
Note: The model path is defined and loaded as `ckpt_dir_model_M4 = 'models_trained_on_cropeed_masks/celeba_bilstm_checkpoint_40.pth.tar'` in `CelebA_syntax/configs/config_lstm.py`.

### Prepare SUNRGBD dataset
SSDI can be trained and evaluated with the [SUN RGB-D Benchmark suit](http://rgbd.cs.princeton.edu/paper.pdf). Please download the data on the [official webpage](http://rgbd.cs.princeton.edu), unzip it, and place it with a folder tree like this,
```bash
YOUR_PATH # Some arbitrary path
├── SUNRGBD # The unzip folder of SUNRGBD.zip
└── SUNRGBDtoolbox # The unzip folder of SUNRGBDtoolbox.zip
```
The root path `SOMEPATH` should be passed to the program using the `--data-dir SOMEPATH` argument.  

Then, in `SUNRGBD_syntax/test_lstm_13_for_paper.sh`, change `--data-dir` to `YOUR_PATH`.

### Prepare SUNRGBD model weights:  
- Download Resnet-50 Encoder-Decoder FPN Segmentation model from [RedNet GDrive Link](https://drive.google.com/uc?export=download&id=1K0vEzCtPh0Eb3VeB_jhatAOLPBQ7FJi5), and put the file under `SUNRGBD_syntax/models_trained_on_masks`.  
Note: The model path is defined and loaded as `--rednet-ckpt models_trained_on_masks/rednet_ckpt.pth` in `SUNRGBD_syntax/test_lstm_13_for_paper.sh`.  
- Download pretrained Bi-LSTM model from [SUNRGBD, Patch Size 80, Bi-LSTM GDrive Link](https://drive.google.com/uc?export=download&id=1i0bQRJ0PEikRoTlAFvj0FMlcpjHyrlHl) and [SUNRGBD, Patch Size 160, Bi-LSTM GDrive Link](https://drive.google.com/uc?export=download&id=11vq2GjOdY7TmXYEPVI0UZ1p6Qwut1ZBi), and put the 2 files under `SUNRGBD_syntax/models_trained_on_masks`.
Note: The model path is defined and loaded as `ckpt_dir_model_M4 = "models_trained_on_masks/sunrgbd_bilstm_input_size=640x480_num_classes={}_ps={}_numlayers=1_startlr=0.0001_checkpoint_40.pth.tar".format(semantics_dim, patch_size)` in `SUNRGBD_syntax/config_lstm.py`.

## Running SSDI Detection
The composition of subfolders in `SSDI` is as follows:
```
- SSDI
  - CelebA_syntax
  - PiCIE-CelebA
  - SSDI_rebuttal
  - SUNRGBD_syntax
  readme.md
```

### Stage One: Learning Part Semantics
To learn part semantics, we first obtain semantic clusters the PiCIE technique, and then finetune a segmentation model on the segmentation masks based on these semantic clusters.  
Scipts to train the self-supervised clustering, retrieve the clusters, and finetune the FPN segementation model on the semantic clusters can be found in the folder `PiCIE-CelebA`.

### Stage Two: Learning Part Syntax
To learn part semantics, we use a Bi-directional LSTM. Scripts to train and test the Bi-LSTM models on CelebA and SUNRGBD are found in `CelebA_syntax` and `SUNRGBD_syntax`, respectively.

#### SSDI Attack Generation
To generate SSDI attacks for images in:
- CelebA dataset:   
  Navigate to `CelebA_syntax/configs/config_test_LSTM.py` and set the variables under "Dataset corruption configs".  
  For example, to apply the "landmark_shuffle" corruption, with a degree of 3 20x20 patches around facial landmarks (5 landmarks in total for each face image), set the variables as follows:
  ```python
  add_corruption              = True
  all_corrupt                 = False # set to True only for puzzle solving task
  landmark_shuffle            = True # always set to False when training
  black_box                   = False
  gaussian_blur               = False
  puzzle_solving              = False
  puzzle_solving_all_perms    = False
  ......
  num_distortion              = 3
  box_size                    = 20
  ```
  * For "landmark_shuffle", set `landmark_shuffle` to `True`, and set the number of patches and their size using `num_distortion` and `box_size`
  * For "black_box", set `black_box` to `True`, and set the number of patches and their size using `num_box` and `box_size`.
  * For "gaussian_blur", set `gaussian_blur` to `True`, and set the number of patches and their size using `num_box` and `box_size`
  * For "puzzle_solving", set `puzzle_solving` to `True`, and set the number of puzzles and the number of patches within each puzzle to be permuted (max 5) and their size using `num_permute` and `box_size`
  * If you want to generate all permutations (5!=120 puzzles in total) of the puzzle solving task, set `puzzle_solving_all_perms` and `puzzle_solving` to `True`, and permuted patch size using `box_size`  

  These variables are used when creating attacked CelebA dataset defined in `CelebA_inthewild` class in `CelebA_syntax/dataloader_celebA_inthewild.py`.

- SUNRGBD dataset:  
  Navigate to `SUNRGBD_syntax/configs/config_LSTM.py` and set the variables under "Dataset corruption configs".  
  For example, to apply the "puzzle_solving" corruption, with a degree of 3 permuted puzzles each containing all-permuted 80x80 patches, set the variables as follows:
  ```python
    all_corrupt = True # only when doing "puzzle solving"
    corruption_type = "puzzle_solving" # choose among ["patch_shuffling", "puzzle_solving", "black_box", "gaussian_blurring"]
    patch_size = 80 # 160 or 80 when input image is 640x480
    num_permute = 3 
  ```
  * For "landmark_shuffle", set `corruption_type` to `landmark_shuffle`, and set the number of patches and their size using `num_distortion` and `patch_size`
  * For "black_box", set `corruption_type` to `black_box`, and set the number of patches and their size using `num_box` and `patch_size`
  * For "gaussian_blur", set `corruption_type` to `gaussian_blur`, and set the number of patches and their size using `num_distortion` and `patch_size`
  * For "puzzle_solving", set `corruption_type` to `puzzle_solving`, and set the number of puzzles and the permuted patch size using `num_permute` and `patch_size`. Each puzzle contains all-permuted 80x80 or 160x160 patches.

  These variables are used when creating attacked CelebA dataset defined in `CelebA_inthewild` class in `CelebA_syntax/dataloader_celebA_inthewild.py`.

#### SSDI Detection Inference
Below are the instructions to run SSDI inference based on the pretrained semantic segmentation models and bi-LSTM models, you would need to refer to `CelebA_syntax` and `SUNRGBD_syntax` folders.

- CelebA  
  There are 3 methods to run image grammar inference on CelebA images, as described in the [Towards Image Semantics and Syntax Sequence Learning](https://arxiv.org/pdf/2401.17515) paper, Figure 4.

  Navigate to the `CelebA_syntax` folder:
  ```bash
  cd CelebA_syntax
  ```
  There are by default 7 part semantics in CelebA images, the number of hyperclusters is set to 20.

  To set configuration for your one-time inference, navigate to `CelebA_syntax/config/config_test_LSTM.py` and change the dataset corruption configs, to apply the desired corruption to the CelebA images. Choices are: `['landmark_shuffle', 'black_box', 'gaussian_blur', 'puzzle_solving']`. You can also determine the degree of corruption.

  Run `bash scripts/bash test_lstm_for_paper.sh` to use method1: Bi-LSTM + next semantics  
  Run `bash scripts/test_lstm_for_paper_using_avg_semantics.sh` to use method2: Bi-LSTM + avg. semantics  
  Run `bash scripts/test_lstm_for_paper_using_avg_masks.sh` to use method3: mIoU with avg. semantics

- SUN-RGBD  
  Navigate to the `SUNRGBD_syntax` folder:
  ```bash
  cd SUNRGBD_syntax
  ```
  There are by default 13 part semantics in SUN-RGBD images, the number of hyperclusters is set to 37.

  To set configuration for your one-time inference, navigate to the `SUNRGBD_syntax/config_lstm.py` file and change the dataset corruption configs, to apply the desired corruption to the CelebA images. Choices are: `['landmark_shuffle', 'black_box', 'gaussian_blur', 'puzzle_solving']`. You can also determine the degree of corruption.

  Run `bash test_lstm_13_for_paper.sh` to use method1: Bi-LSTM + next semantics  
  Run `bash test_lstm_13_for_paper_using_avg_semantics.sh` to use method2: Bi-LSTM + avg. semantics  
  Run `bash test_lstm_13_for_paper_using_avg_masks.sh` to use method3: mIoU with avg. semantics

## SSDI Detection Results
| Dataset | Part Semantics Model | Grammar Validation Method | Shuffle 4 160x160 | Shuffle 16 80x80 | Black 4 160x160 | Black 16 80x80 |
|---------|----------------------|---------------------------|-------------------|-----------------|-----------------|----------------|
| SUN-RGBD (13-cls.) | ResNet50+Encoder-Decoder (ours) | Bi-LSTM + next semantics | 60.57 (72.04) | 76.57 (73.37) | 66.55 (67.21) | 67.43 (65.53) |
| SUN-RGBD (13-cls.) | Dformer-S (Yin et al.) (2023) | Bi-LSTM + next semantics	| 54.97 (65.36)	| 61.52 (64.09)	| 58.46 (59.65)	| 61.92 (62.05) |
| SUN-RGBD (13-cls.) | GroundingDino + GroundedSAM2 (2024) | Bi-LSTM + next semantics | **92.01 (86.27)** | 74.60 (61.80) | **93.30 (88.43)**	| 73.02 (59.91) |

\* first number is detection accuracy; second number in parenthesis is detection rate, meaured based on best_score = 0.5 * num_tn + 0.5 * num_tp (equal weight to true-negative and true-positive samples)  

More SSDI results, including T-SNE visualization of CelebA and SUNRGBD semantic clusters, can be found in the `SSDI_rebuttal` folder.

## Acknowledgments
The SSDI work uses 2 datasets, [CelebAMask-HQ Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) and [SUN-RGBD](https://rgbd.cs.princeton.edu/).  
This project utilizes the [PiCIE](https://github.com/janghyuncho/PiCIE) framework for unsupervised image semantic representation learning.    
We referred to the code in [RedNet](https://github.com/JindongJiang/RedNet) for the semantic segmentation model traning and inference on SUNRGBD dataset.  
Additionally, we generated SSDI detection results using [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2): Ground and Track Anything in Videos.  
Thanks to the authors of above repositories for their work.

## Citation
If you find SSDI useful in your research, please consider citing:
```
@article{
tao2025semanticsyntactic,
title={Semantic-Syntactic Discrepancy in Images ({SSDI}): Learning Meaning and Order of Features from Natural Images},
author={Chun Tao and Timur Ibrayev and Kaushik Roy},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=8otbGorZK2},
note={}
}
```
Or the [preprint version](https://arxiv.org/abs/2401.17515).
