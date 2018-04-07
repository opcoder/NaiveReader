#!/bin/bash
export PYTHONUSERBASE=/mnt/lustre/wangchangbao/.local
#source /mnt/lustre/wangchangbao/py3tf1.6/bin/activate
source /mnt/lustre/share/wangchangbao/workspace/py2tf1.6/bin/activate
export LD_LIBRARY_PATH=/mnt/lustre/share/nccl_2.1.15-1+cuda9.0_x86_64:/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/lustre/share/wangchangbao/workspace/NaiveReader/utils:/mnt/lustre/share/wangchangbao/workspace/NaiveReader/tensorflow:$PYTHONPATH

root=/mnt/lustre/share/wangchangbao/workspace/NaiveReader/data
datadir=${root}/datasets/preprocessed
prefix=search
#prefix=zhidao
algo=BIDAF
name=$1
gpu=$2
model_dir=${root}/${prefix}/models/models_${name}/${algo}
mkdir -p ${model_dir}
srun -p DSK \
    --gres=gpu:${gpu} -n1 --ntasks-per-node=1 \
    --job-name=${name}-${gpu} \
    --kill-on-bad-exit=1 \
    python run.py --predict --algo ${algo} \
  --model_dir=${model_dir} \
  --epochs=10 \
  --decay_epochs=20 \
  --min_cnt=2 \
  --vocab_dir=${root}/${prefix}/vocab \
  --result_dir=${root}/${prefix}/results \
  --summary_dir=${root}/${prefix}/summary \
  --train_files=${datadir}/trainset/${prefix}.train.json \
  --test_files=${datadir}/testset/${prefix}.test.json \
  --dev_files=${datadir}/devset/${prefix}.dev.json \
  --learn_word_embedding=1 \
  2>&1 | tee ${model_dir}/predict.log &
