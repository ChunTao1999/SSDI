K_train=7
K_test=7
bsize=128
num_epoch=20
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4

mkdir -p results/picie/train/${SEED}

# flag unbuffer for pdb purpose
# CUDA_VISIBLE_DEVICES=0 
# --pretrain \
# --restart \

python -u finetune_for_fine_features.py \
--celeba \
--data_root /home/nano01/a/tao88/celebA_raw/CelebAMask-HQ \
--save_root finetuned_models/finetuned_models_coarse/train/${SEED} \
--pretrain \
--repeats 1 \
--lr ${LR} \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--stuff --thing  \
--batch_size_cluster ${bsize} \
--num_epoch ${num_epoch} \
--res 256 --res1 256 --res2 512 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip \
--finetune # important

