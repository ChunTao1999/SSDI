## Introduction to the Image Grammar Project repo
The Image Grammar project provides a new way to learn the visual semantics and syntax, with respect to a class of objects (e.g., face in CelebA and CelebAHQ) and a class of scenes (e.g., rooms in SUN-RGBD and SUN-RGBD-13-Classes).

The aim of this readme file is to allow anyone to reproduce code results of the above project.

## Rebuttal
The rebuttal folder is at `SSDI_rebuttal/`, model checkpoints [here](https://drive.google.com/drive/folders/1-2pD-6OPCCL-AzsxBTiGQ4_5if_4C4OA?usp=sharing).
To run the rebuttal notebook file `SSDI_rebuttal/train_lstm_sunrgbd.ipynb`, please
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
under the `SSDI_rebuttal` folder.  

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