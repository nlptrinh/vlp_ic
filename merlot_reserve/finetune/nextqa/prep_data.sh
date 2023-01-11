#!/usr/bin/env bash

export NUM_FOLDS=64
export NUM_FOLDS_VAL=8

# mkdir -p logs_add_img_encoded
# cd logs_add_img_encoded
python prep_data.py -next_qa_file 'txt_nextqa_blip_clip_loss_whole.json'  -num_folds ${NUM_FOLDS} ../../pretrain/configs/base.yaml 'train' > logs_add_img_encoded/trainlog.txt
# python prep_data.py -next_qa_file 'txt_nextqa_blip_clip_loss_whole.json' -num_folds ${NUM_FOLDS_VAL} ../../pretrain/configs/base.yaml  'val' > logs_add_img_encoded/vallog.txt

# parallel -j $(nproc --all) --will-cite "python prep_data.py -fold {1} -num_folds ${NUM_FOLDS} > logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))
# parallel -j $(nproc --all) --will-cite "python prep_data.py -fold {1} -num_folds ${NUM_FOLDS_VAL} -split=val > logs/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS_VAL}-1)))
# parallel -j $(nproc --all) --will-cite "python prep_data.py -fold {1} -num_folds ${NUM_FOLDS_VAL} -split=test > logs/testlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS_VAL}-1)))

