#!/bin/bash

root=../
export PYTHONPATH=${root}/utils:${root}/tensorflow:$PYTHONPATH

datadir=/mnt/lustre/share/wangchangbao/datasets/MRC/DuReader/preprocessed/
algo=BIDAF
name=$1
gpu=$2
exp_name=${name}-${gpu}
model_dir=${exp_name}
mkdir -p ${model_dir}
srun -p DSK \
    --gres=gpu:${gpu} -n1 --ntasks-per-node=1 \
    --job-name=${exp_name} \
    --kill-on-bad-exit=1 \
    python run.py --evaluate --algo ${algo} \
  --model_dir=${model_dir} \
  --epochs=10 \
  --decay_epochs=20 \
  --min_cnt=2 \
  --vocab_dir=${root}/${prefix}/vocab \
  --result_dir=${root}/${prefix}/results \
  --summary_dir=${root}/${prefix}/summary \
  --train_files=${datadir}/trainset/${prefix}.train.json \
  --test_files=${datadir}/testset/${prefix}.test.json \
  --dev_files=${datadir}/devset/${prefix}.dev.json.selected \
  --learn_word_embedding=1 \
  2>&1 | tee ${model_dir}/train.log &
