in_dim=128
K_train=7
K_test=7
bsize=256
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4

# mkdir -p results/picie/train/${SEED}

# flag unbuffer for pdb purpose
# CUDA_VISIBLE_DEVICES=1

python -u visualize_glimpse_seg_masks.py \
--num_epoch ${num_epoch} \
--pretrain \
--FPN_with_classifier \
--in_dim ${in_dim} \
--K_train ${K_train} \
--eval_only \
--eval_path /home/nano01/a/tao88/foveation_grammar_detection/glimpse_classification_models/checkpoint_20.pth.tar \
--lr ${LR} \
--seed ${SEED} \
