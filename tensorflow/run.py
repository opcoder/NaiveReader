# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module prepares and runs the whole system.
"""

import os
#import importlib
import sys
#importlib.reload(sys)
#sys.setdefaultencoding('utf8')
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
sys.path.append('../utils')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import log_helper
log_helper.init_log('brc')

import pickle
import argparse
import logging
from dataset import BRCDataset
from vocab import Vocab
from rc_model import RCModel
from model_helper_v1 import ModelHelper
import gensim

logger = logging.getLogger("brc")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--decay_epochs', type=int, default=3,
                                help='epochs to decay')
    train_settings.add_argument('--decay_rate', type=float, default=0.1,
                                help='rate to decay')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--char_embed_size', type=int, default=300,
                                help='size of the char_embeddings')
    model_settings.add_argument('--use_char_embed', type=bool, default=False,
                                help='wether use char_embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')
    model_settings.add_argument('--word2vec_path', type=str, default='',
                                help='path to word2vec')
    model_settings.add_argument('--learn_word_embedding', type=str2bool, default='1',
                                help='wether to learn word embedding or not')
    
    model_settings.add_argument('--min_cnt', type=int, default=2,
                                help='minimum count of valid word')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    if args.use_char_embed:
        char_vocab = Vocab(lower=True)
    fout = open(os.path.join(args.vocab_dir, 'word.txt'), 'w')
    for word in brc_data.word_iter('train'):
        if word == 'mosi':
            print('xxxxxxxxxx:%s\n' % word)
        vocab.add(word)
        if args.use_char_embed:
            for char in list(word):
                char_vocab.add(char)
        fout.write(word + ', ' + ' '.join(list(word)) + '\n')
    fout.close()
    idx = vocab.get_id('mosi')
    print('yyyyyyyy:mosi_idx:%d\n' % idx)


    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=args.min_cnt)
    logger.info('min_cnt = %d ' % args.min_cnt)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    
    #logger.info('The final char vocab size is {}'.format(char_vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)
    if args.use_char_embed:
        char_vocab.randomly_init_embeddings(args.char_embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)
    
    if args.use_char_embed:
        logger.info('Saving char vocab...')
        with open(os.path.join(args.vocab_dir, 'char_vocab.data'), 'wb') as fout:
            pickle.dump(char_vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    if args.word2vec_path:
        logger.info('learn_word_embedding:{}'.format(args.learn_word_embedding))
        logger.info('loadding %s \n' % args.word2vec_path)
        word2vec = gensim.models.Word2Vec.load(args.word2vec_path)
        vocab.load_pretrained_embeddings_from_w2v(word2vec.wv)
        logger.info('load pretrained embedding from %s done\n' % args.word2vec_path)

    if args.use_char_embed:
        with open(os.path.join(args.vocab_dir, 'char_vocab.data'), 'rb') as fin:
            char_vocab = pickle.load(fin)

    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    steps_per_epoch = brc_data.size('train') // args.batch_size
    args.decay_steps = args.decay_epochs * steps_per_epoch 
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    
    if args.use_char_embed:
        logger.info('Converting text into char ids...')
        brc_data.convert_to_char_ids(char_vocab)
        logger.info('Binding char_vocab to args to pass to RCModel')
        args.char_vocab = char_vocab

    RCModel = choose_model_by_gpu_setting(args)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...{}'.format(RCModel.__name__))
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    steps_per_epoch = brc_data.size('dev') // args.batch_size
    args.decay_steps = args.decay_epochs * steps_per_epoch 
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    RCModel = choose_model_by_gpu_setting(args)
    rc_model = RCModel(vocab, args)
    logger.info('Restoring the model...{}'.format(RCModel.__name__))
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted', save_full_info=False)
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    steps_per_epoch = brc_data.size('train') // args.batch_size
    args.decay_steps = args.decay_epochs * steps_per_epoch 
    RCModel = choose_model_by_gpu_setting(args)
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def choose_model_by_gpu_setting(args):
    model = RCModel
    gpus = eval(os.environ['CUDA_VISIBLE_DEVICES'])
    if isinstance(gpus, int):
        gpus = gpus,
    args.gpus = args.gpu = gpus
    logger.info('gpus:{}'.format(args.gpus))
    if len(gpus) >= 1:
        model = ModelHelper
        logger.info('using parallel training mode')
    else:
        logger.info('using single gpu training mode')
    return model


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    choose_model_by_gpu_setting(args)
    batch_size_per_gpu = args.batch_size
    args.batch_size = args.batch_size * len(args.gpus)
    logger.info('batch-size:{}={}*{}'.format(args.batch_size, batch_size_per_gpu, len(args.gpus)))

    '''
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    '''
    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
