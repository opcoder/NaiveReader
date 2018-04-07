#!/bin/bash
export PYTHONUSERBASE=/mnt/lustre/wangchangbao/.local
#source /mnt/lustre/wangchangbao/py3tf1.6/bin/activate
source /mnt/lustre/share/wangchangbao/workspace/py2tf1.6/bin/activate
export LD_LIBRARY_PATH=/mnt/lustre/share/nccl_2.1.15-1+cuda9.0_x86_64:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/lustre/share/wangchangbao/workspace/NaiveReader/utils:/mnt/lustre/share/wangchangbao/workspace/NaiveReader/tensorflow:$PYTHONPATH

datadir=/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed
dataset=search
#dataset=zhidao
#dataset=combine
algo=BIDAF
model_name=$1
model_root=/mnt/lustre/share/wangchangbao/workspace/NaiveReader
model_dir=${model_root}/models/${dataset}/${model_name}/${algo}
mkdir -p ${model_dir}
srun -p DSK \
    --gres=gpu:$2 -n1 --ntasks-per-node=1 \
    --job-name=mrc$1 \
    --kill-on-bad-exit=1 \
    python run.py --prepare --algo ${algo} \
  --learning_rate=0.001 \
  --model_dir=${model_dir} \
  --epochs=10 \
  --decay_epochs=5 \
  --min_cnt=2 \
  --vocab_dir=${model_root}/${dataset}/vocab \
  --result_dir=${model_dir}/results \
  --summary_dir=${model_dir}/summary \
  --train_files=${datadir}/trainset/${dataset}.train.json \
  --test_files=${datadir}/test1set/${dataset}.test1.json \
  --dev_files=${datadir}/devset/${dataset}.dev.json \
  --learn_word_embedding=1 \
  --batch_size=32 \
  2>&1 | tee ${model_dir}/train.log &
