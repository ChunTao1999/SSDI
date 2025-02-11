## Introduction to the Image Grammar Project repo
The Image Grammar project provides a new way to learn the visual semantics and syntax, with respect to a class of objects (e.g., face in CelebA and CelebAHQ) and a class of scenes (e.g., rooms in SUN-RGBD and SUN-RGBD-13-Classes).

The aim of this readme file is to allow anyone to reproduce code results of the above project.

## Rebuttal
The rebuttal folder is at `SSDI_rebuttal/`. To run the rebuttal notebook file `SSDI_rebuttal/train_lstm_sunrgbd.ipynb`, please first clone the Grounded-SAM-2 repo:
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
under the `SSDI_rebuttal` folder.  

### Install GroundingDino and Grounded-SAM2
Please follow instructions [here](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file#installation) to install the GroundingDino and Grounded-SAM2 repos and download the model weights.

### Download the SUNRGBD dataset and train and test set segmentation masks
[SUNRGBD train dataset](https://drive.google.com/drive/folders/1ZGnoKfgYjfUpU-02-kS8hi2MZzuCAqfz?usp=sharing)  
[SUNRGBD train dataset with bounding boxes and segmentation](https://drive.google.com/drive/folders/1LIw5QnPsayLw5t6hgK8P_JgDae8-gpWG?usp=sharing)  
[SUNRGBD train dataset 13-class segmentation masks](https://drive.google.com/drive/folders/1ZRAxYs4e2qeiZtiejVGkGTaGlYLwcEkQ?usp=sharing)  

[SUNRGBD test dataset](https://drive.google.com/drive/folders/1FeGAG0ZyvkK2RaeGWXnKYc_l9znAnjSv?usp=sharing)  
[SUNRGBD test dataset with bounding boxes and segmentation](https://drive.google.com/drive/folders/1ZbdaK1CnQ1hf-YurPuHZ39Xg1V3IYTme?usp=sharing)  
[SUNRGBD test dataset 13-class segmentation masks](https://drive.google.com/drive/folders/1oLkQ7U8dhb-MdAO6fdeBP6qa7B5pd2Yu?usp=sharing)  


### Bi-LSTM model checkpoints
Please run `SSDI_rebuttal/train_lstm_sunrgbd.ipynb` to train the Bi-LSTM models.  
The Bi-LSTM models are trained on the SUNRGBD segmentation masks, 14 object classes including background, generated from GroundingDino and GroundedSAM2. The model checkpoints are available [here](https://drive.google.com/drive/folders/1uPyaEZRKzgUbcAYvdOSNwKgqeenkDvTy?usp=sharing).

### Rebuttal results
Please find the results below or at `SSDI/SSDI_rebuttal/inference_results.xlsx`  
| Dataset | Part Semantics Model | Grammar Validation Method | Shuffle 4 160x160 | Shuffle 16 80x80 | Black 4 160x160 | Black 16 80x80 |
|---------|----------------------|---------------------------|-------------------|-----------------|-----------------|----------------|
| SUN-RGBD (13-cls.) | ResNet50+Encoder-Decoder (ours) | Bi-LSTM + next semantics | 60.57 (72.04) | 76.57 (73.37) | 66.55 (67.21) | 67.43 (65.53) |
| SUN-RGBD (13-cls.) | Dformer-S (Yin et al.) (2023) | Bi-LSTM + next semantics	| 54.97 (65.36)	| 61.52 (64.09)	| 58.46 (59.65)	| 61.92 (62.05) |
| SUN-RGBD (13-cls.) | GroundingDino + GroundedSAM2 (2024) | Bi-LSTM + next semantics | **92.01 (86.27)** | 74.60 (61.80) | **93.30 (88.43)**	| 73.02 (59.91) |

\* first number is detection accuracy; second number in parenthesis is detection rate, meaured based on best_score = 0.5 * num_tn + 0.5 * num_tp (equal weight to true-negative and true-positive samples)



### Past rebuttal
Another rebuttal file is at `cvpr24_image_semantics/foveation_grammar_detection_celebA/rebuttal.py`



## Organization of folders
The folders needed to run everything of the Image Grammar project are `cvpr2024_submission` and `cvpr24_image_semantics`. 

All important scripts can be found in `cvpr2024_submission`. CNN and LSTM model weights are stored in `cvpr24_image_semantics`, and these weights are loaded through the scripts in `cvpr2024_submission`.

The composition of subfolders in `cvpr2024_submission` is as follows:
```
- cvpr2024_submission
  - CelebA_syntax
  - grammar_vit
  - part_semantics
  - rebuttal
  - SUNRGBD_syntax
  readme.md
```
`CelebA_syntax` contains train and test scripts for validating grammar of normal and corrupted CelebA images;  
`SUNRGBD_syntax` contains train and test scripts for validating grammar of normal and corrupted SUN-RGBD images;  
`grammar_vit` contains test scripts for validating grammar of normal and corrupted CelebA images using the ViT model;  
`part_semantics` contains fine-tuning scripts for training feature extractor CNNs on segmented image windows and corresponding segmentation masks;
`rebuttal` contains T-SNE rebuttal images;

## Run inference
The run inference for the pretrained bi-LSTM models, you would need to refer to the `cvpr2024_submission/CelebA_syntax` and `cvpr2024_submission/SUNRGBD_syntax` folders. 

### CelebA
There are 3 methods to run image grammar inference on CelebA images, as described in the [Towards Image Semantics and Syntax Sequence Learning](https://arxiv.org/pdf/2401.17515) paper, Figure 4.

Navigate to the `cvpr2024_submission/CelebA_syntax` folder:
```bash
cd cvpr2024_submission/CelebA_syntax
```

There are by default 7 part semantics in CelebA images, the number of hyperclusters is 20.

To set configuration for your one-time inference, navigate to the `config/config_test_LSTM.py` file and change the dataset corruption configs, to apply the desired corruption to the CelebA images. Choices are: `['landmark_shuffle', 'black_box', 'gaussian_blur', 'puzzle_solving']`. You can also determine the degree of corruption.

Run `bash scripts/bash test_lstm_for_paper.sh` to use method1: Bi-LSTM + next semantics  
Run `bash scripts/test_lstm_for_paper_using_avg_semantics.sh` to use method2: Bi-LSTM + avg. semantics  
Run `bash scripts/test_lstm_for_paper_using_avg_masks.sh` to use method3: mIoU with avg. semantics


### SUN-RGBD
There are by default 13 part semantics in SUN-RGBD images, the number of hyperclusters is 37.

To set configuration for your one-time inference, navigate to the `SUNRGBD_syntax/config_lstm.py` file and change the dataset corruption configs, to apply the desired corruption to the CelebA images. Choices are: `['landmark_shuffle', 'black_box', 'gaussian_blur', 'puzzle_solving']`. You can also determine the degree of corruption.

Run `bash test_lstm_13_for_paper.sh` to use method1: Bi-LSTM + next semantics  
Run `bash test_lstm_13_for_paper_using_avg_semantics.sh` to use method2: Bi-LSTM + avg. semantics  
Run `bash test_lstm_13_for_paper_using_avg_masks.sh` to use method3: mIoU with avg. semantics

## Walkthrough of scripts
### Scripts for generating attacks
### Scripts for loading data (dataloader scripts)
### Scripts for loading pre-trained models