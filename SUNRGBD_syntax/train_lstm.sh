BATCH_SIZE_TRAIN=2
NUM_WORKERS=8
OUTPUT_DIR='/home/nano01/a/tao88/4.26_train_lstm_640_480_ps=80'

python -u train_lstm.py \
--cuda \
--data-dir /home/nano01/a/tao88/SUN-RGBD \
-b ${BATCH_SIZE_TRAIN} \
-j ${NUM_WORKERS} \
-o ${OUTPUT_DIR} \
--rednet-ckpt ./pretrained/rednet_ckpt.pth