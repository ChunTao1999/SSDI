K_train=20
K_test=20
bsize=128
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=100
SEED=1
LR=1e-4

mkdir -p results/picie/train/${SEED}

# flag unbuffer for pdb purpose
# CUDA_VISIBLE_DEVICES=0 
# --eval_only \

python -u play_tensorboard.py \
--celeba \
--data_root /home/nano01/a/tao88/celebA_raw \
--save_root results/picie_on_celeba/train/${SEED} \
--pretrain \
--repeats 1 \
--finetuned_model_path /home/nano01/a/tao88/PiCIE-CelebA/finetuned_models/finetuned_models_coarse/train/1finetune/augmented/res1=256_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale\\=0.5/K_train=7_cosine/checkpoint_20_epochs.pth.tar \
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
--model_finetuned
