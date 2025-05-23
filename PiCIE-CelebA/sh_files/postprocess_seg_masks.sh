K_train=20
K_test=20
bsize=256
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4

mkdir -p results/picie/train/${SEED}

# flag unbuffer for pdb purpose
# CUDA_VISIBLE_DEVICES=0 
# use the 8th epoch model params for better results

python -u postprocess_seg_masks.py \
--celeba \
--data_root /home/nano01/a/tao88/celebA_raw \
--save_root results/picie_on_celeba/train/${SEED} \
--pretrain \
--repeats 1 \
--eval_only \
--eval_path /home/nano01/a/tao88/PiCIE-CelebA/results/picie_on_celeba_finetuned_model_coarse/train/1/augmented/res1=256_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale\\=0.5/K_train=20_cosine/checkpoint_7.pth.tar \
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
--with_mask
