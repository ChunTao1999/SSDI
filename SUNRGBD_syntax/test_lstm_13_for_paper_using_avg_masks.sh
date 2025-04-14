BATCH_SIZE_TEST=32
NUM_WORKERS=8
OUTPUT_DIR='./11.7/sun_13_patch_160/transition_demo'

python -u test_lstm_13_for_paper_using_avg_masks.py \
--cuda \
--data-dir /home/nano01/a/tao88/SUN-RGBD \
--output-dir ${OUTPUT_DIR} \
-b ${BATCH_SIZE_TEST} \
-j ${NUM_WORKERS} \
--rednet-ckpt ./pretrained/rednet_ckpt.pth

