#!/bin/bash
datadir=/data2/changbao/MCdataset/DuReader/DuReader/data/preprocessed
#datadir=/data2/changbao/MCdataset/DuReader/DuReader/data/raw
prefix=search
#prefix=zhidao
model=$1
gpu_id=$2
model_dir=../data/models_char/${model}
python run.py --train --algo ${model} --gpu ${gpu_id} \
  --model_dir=${model_dir} \
  --epochs=10 \
  --train_files=${datadir}/trainset/${prefix}.train.json \
  --test_files=${datadir}/testset/${prefix}.test.json \
  --dev_files=${datadir}/devset/${prefix}.dev.json \
  2>&1 | tee train_char_${model}.log &
