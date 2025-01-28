in_dim=128
K_train=7
K_test=7
bsize=256
num_epoch=100
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4

# mkdir -p results/picie/train/${SEED}

# flag unbuffer for pdb purpose
# CUDA_VISIBLE_DEVICES=1
# --eval_path /home/nano01/a/tao88/PiCIE-CelebA/results/picie_on_celeba_finetuned_model_coarse/train/1/augmented/res1=256_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale\\=0.5/K_train=20_cosine/checkpoint_7.pth.tar \
# --eval_path /home/nano01/a/tao88/foveation_grammar_detection/glimpse_classification_models/checkpoint.pth.tar \

python -u train_on_crop_seg_masks.py \
--num_epoch ${num_epoch} \
--pretrain \
--FPN_with_classifier \
--in_dim ${in_dim} \
--K_train ${K_train} \
--eval_only \
--eval_path /home/nano01/a/tao88/PiCIE-CelebA/results/picie_on_celeba_finetuned_model_coarse/train/1/augmented/res1=256_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale\\=0.5/K_train=20_cosine/checkpoint_7.pth.tar \
--lr ${LR} \
--seed ${SEED} \
