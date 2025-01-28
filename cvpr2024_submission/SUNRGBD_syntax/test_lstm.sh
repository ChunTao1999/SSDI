BATCH_SIZE_TEST=32
NUM_WORKERS=8
OUTPUT_DIR='/home/nano01/a/tao88/4.24_test_lstm_640_480/corrupted'

python -u test_lstm.py \
--cuda \
--data-dir /home/nano01/a/tao88/SUN-RGBD \
--output-dir ${OUTPUT_DIR} \
-b ${BATCH_SIZE_TEST} \
-j ${NUM_WORKERS} \
--rednet-ckpt ./pretrained/rednet_ckpt.pth

