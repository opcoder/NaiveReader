#!/bin/bash
export PYTHONUSERBASE=/mnt/lustre/wangchangbao/.local
#source /mnt/lustre/wangchangbao/py3tf1.6/bin/activate
source /mnt/lustre/wangchangbao/MCdataset/py2tf1.6/bin/activate
export LD_LIBRARY_PATH=/mnt/lustre/share/nccl_2.1.2-1+cuda9.0_x86_64/lib:/mnt/lustre/share/yufengwei/conda_envs/v100/lib:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH

datadir=/mnt/lustre/wangchangbao/MCdataset/DuReader/DuReader/data/preprocessed
#datadir=/data2/changbao/MCdataset/DuReader/DuReader/data/raw
prefix=search
#prefix=zhidao
model=$1
model_dir=../data/models_$2/${model}
mkdir -p ${model_dir}
srun -p V100 \
    --gres=gpu:1 -n1 --ntasks-per-node=1 \
    --job-name=dr$2 \
    --kill-on-bad-exit=1 \
    python run.py --train --algo ${model} \
  --model_dir=${model_dir} \
  --epochs=10 \
  --min_cnt=2 \
  --train_files=${datadir}/trainset/${prefix}.train.json \
  --test_files=${datadir}/testset/${prefix}.test.json \
  --dev_files=${datadir}/devset/${prefix}.dev.json \
  --learn_word_embedding=1 \
  2>&1 | tee ${model_dir}/train.log &
