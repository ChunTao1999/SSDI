IMG_PATH=/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-img
LABEL_PATH=/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-label-coarse
LABEL_MAP_PATH=/home/nano01/a/tao88/cvpr24_image_semantics_clean/celebAHQ/celebahq-id2label.json
BS_TRAIN=128

python3 -u train_vit_on_celeba.py \
--celebahq_img_path ${IMG_PATH} \
--celebahq_label_path ${LABEL_PATH} \
--label_mapping_path ${LABEL_MAP_PATH} \
--batch_size_train ${BS_TRAIN}

