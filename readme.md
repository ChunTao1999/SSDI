## Introduction to SSDI
This repository provides the code and resources for the Image Grammar project, which implements the methods described in the [paper](https://arxiv.org/abs/2401.17515) titled *"Semantic-Syntactic Discrepancy in Images (SSDI): Learning Meaning and Order of Features from Natural Images"*. It includes tools to reproduce the results presented in the paper, focusing on the proposed two-stage semantic and syntactic learning method and SSDI detection. 

The project addresses discrepancies in images within specific object classes (e.g., faces in CelebA and CelebAHQ) and scene classes (e.g., rooms in SUN-RGBD and SUN-RGBD-13-Classes). 

With this repository, you can:
- Generate SSDI attacks for images in the CelebA and SUN-RGBD datasets
- Train and run inference using the two-stage PiCIE and bi-LSTM models for SSDI learning
- Reproduce SSDI detection results for the CelebA and SUN-RGBD datasets

## Setup
Set-ups for this project involve the preparing datasets and downloading the pretrained model weights
### Stage One: Learning Part Semantics
### Stage Two: Learning Part Syntax

### SSDI Detection
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

- SUNRGBD dataset
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
To run inference for SSDI detection, navigate to 


## Acknowledgments
This project utilizes the [PiCIE](https://github.com/janghyuncho/PiCIE) framework for unsupervised image semantic representation learning. If you use PiCIE in your work, please consider citing their paper.

## Citation
If you find SSDI useful in your research, please consider citing:
```
@article{
    anonymous2024semanticsyntactic,
    title={Semantic-Syntactic Discrepancy in Images ({SSDI}): Learning Meaning and Order of Features from Natural Images},
    author={Anonymous},
    journal={Submitted to Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=8otbGorZK2},
    note={Under review}
}
```
Or the [preprint version](https://arxiv.org/abs/2401.17515).
