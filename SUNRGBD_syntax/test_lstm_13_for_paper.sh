BATCH_SIZE_TEST=32
NUM_WORKERS=8
OUTPUT_DIR='./transition_demo'

python -u test_lstm_13_for_paper.py \
--cuda \
--data-dir /home/nano01/a/tao88/SUN-RGBD \
--output-dir ${OUTPUT_DIR} \
-b ${BATCH_SIZE_TEST} \
-j ${NUM_WORKERS} \
--rednet-ckpt models_trained_on_masks/rednet_ckpt.pth

