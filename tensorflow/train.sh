#!/bin/bash
root=..
export PYTHONPATH=${root}/utils:${root}/tensorflow:$PYTHONPATH

datadir=/mnt/lustre/share/wangchangbao/datasets/MRC/dureaderv2.0/preprocessed
search_train=${datadir}/trainset/search.train.json
zhidao_train=${datadir}/trainset/zhidao.train.json
search_dev=${datadir}/devset/search.dev.json
zhidao_dev=${datadir}/devset/zhidao.dev.json
search_test=${datadir}/testset/search.test.json
zhidao_test=${datadir}/testset/zhidao.test.json

algo=BIDAF
model_name=$1
model_root=.
model_dir=${model_root}/modelsv2.0/${model_name}
mkdir -p ${model_dir}
srun -p Spring_face \
    --gres=gpu:$2 -n1 --ntasks-per-node=1 \
    --job-name=$1 \
    --kill-on-bad-exit=1 \
    python run.py --train --algo ${algo} \
  --evaluate \
  --learning_rate=0.001 \
  --model_dir=${model_dir} \
  --epochs=10 \
  --decay_epochs=5 \
  --min_cnt=2 \
  --vocab_dir=${model_root}/vocab \
  --result_dir=${model_dir}/results \
  --summary_dir=${model_dir}/summary \
  --train_files ${search_train} ${zhidao_train} \
  --test_files ${search_test} ${zhidao_test} \
  --dev_files ${search_dev} ${zhidao_dev} \
  --learn_word_embedding=1 \
  --batch_size=32 \
  2>&1 | tee ${model_dir}/train.log 
