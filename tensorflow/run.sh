#!/bin/bash
export PYTHONUSERBASE=/mnt/lustre/wangchangbao/.local
#source /mnt/lustre/wangchangbao/py3tf1.6/bin/activate
source /mnt/lustre/wangchangbao/MCdataset/py2tf1.6/bin/activate
export LD_LIBRARY_PATH=/mnt/lustre/share/nccl_2.1.2-1+cuda9.0_x86_64/lib:/mnt/lustre/share/yufengwei/conda_envs/v100/lib:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH

datadir=/mnt/lustre/wangchangbao/MCdataset/DuReader/DuReader/data/preprocessed
#datadir=/data2/changbao/MCdataset/DuReader/DuReader/data/raw
#prefix=search
prefix=zhidao
model=BIDAF
model_dir=../data/${prefix}/models_$1/${model}
mkdir -p ${model_dir}
srun -p V100 \
    --gres=gpu:$2 -n1 --ntasks-per-node=1 \
    --job-name=mrc$1 \
    --kill-on-bad-exit=1 \
    python run.py --train --algo ${model} \
  --learning_rate=0.001 \
  --model_dir=${model_dir} \
  --epochs=30 \
  --decay_epochs=20 \
  --min_cnt=2 \
  --vocab_dir=../data/${prefix}/vocab \
  --result_dir=../data/${prefix}/results \
  --summary_dir=../data/${prefix}/summary \
  --train_files=${datadir}/trainset/${prefix}.train.json \
  --test_files=${datadir}/testset/${prefix}.test.json \
  --dev_files=${datadir}/devset/${prefix}.dev.json \
  --learn_word_embedding=1 \
  --batch_size=64 \
  2>&1 | tee ${model_dir}/train.log &
