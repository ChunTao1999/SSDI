BATCH_SIZE_TRAIN=16
NUM_WORKERS=8
EPOCHS=40

python -u train_rednet.py \
--cuda \
--data-dir /home/nano01/a/tao88/SUN-RGBD \
--epochs ${EPOCHS}
-b ${BATCH_SIZE_TRAIN} \
-j ${NUM_WORKERS} \
--last-ckpt ./pretrained/rednet_ckpt.pth
