from utils import compute_bleu_rouge
from utils import normalize
from parallel_rc_model import RCModel as PRCModel
import tensorflow as tf
import numpy as np
import logging
import time
import os
import json

class ModelHelper:
    def __init__(self, vocab, args):
        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        self.logger = logging.getLogger('brc')
        self.vocab = vocab
        self.optimizer = self.build_optimizer(args)
        self._setup_placeholders()
        self.input_split = self._scatter_input(args)
        self.train_op, self.start_probs, self.end_probs = self.build_train_and_inference_op(args)

        #sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.graph = tf.get_default_graph()
        self.saver = tf.train.Saver(tf.global_variables())
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(args.model_dir, self.graph)
    
    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.start_label_probs = tf.placeholder(tf.float32, [None, None])
        self.end_label_probs = tf.placeholder(tf.float32, [None, None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _scatter_input(self, args):
        num_gpus = len(args.gpus)
        input = {}
        input['passage_tokens'] = tf.split(self.p, num_gpus, axis=0)
        input['question_tokens'] = tf.split(self.q, num_gpus, axis=0)
        input['passage_length'] = tf.split(self.p_length, num_gpus, axis=0)
        input['question_length'] = tf.split(self.q_length, num_gpus, axis=0)
        input['start_label'] = tf.split(self.start_label, num_gpus, axis=0)
        input['end_label'] = tf.split(self.end_label, num_gpus, axis=0)
        input['start_label_probs'] = tf.split(self.start_label_probs, num_gpus, axis=0)
        input['end_label_probs'] = tf.split(self.end_label_probs, num_gpus, axis=0)
        input['dropout_keep_prob'] = self.dropout_keep_prob
        input_split = [{k:v[i] for k, v in input.items()} for i in range(num_gpus)]
        return input_split

    def build_optimizer(self, args):
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.train.exponential_decay(args.learning_rate, self.global_step, decay_steps=args.decay_steps, decay_rate=args.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        return self.optimizer
    
    def average_gradients(self, tower_grads, args):
        average_grads = []
        num_gpus = len(args.gpus)
        for i, grad_and_vars in enumerate(zip(*tower_grads)):
            gpu_id = args.gpus[i % num_gpus]
            with tf.device('/gpu:%d' % gpu_id):
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(grads, 0, name='xxxxxxxxxxxxxxxxxxxxxxxxxxx')
                grad = tf.reduce_mean(grad, 0, name='yyyyyyyyyyyyyyyyy')

                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)

        return average_grads

    
    
    def timestamp(self):
        return tf.py_func(time.time, [], tf.float64)

    def _gather_output(self, tower_predictions):
        start_probs = tf.concat([_[0] for _ in tower_predictions], axis=0)
        end_probs = tf.concat([_[1] for _ in tower_predictions], axis=0)
        return start_probs, end_probs

    def build_train_and_inference_op(self, args):
        tower_grads = []
        tower_predictions = []
        reuse_variables = None 
        total_losses = []
        #self.time0 = self.timestamp()
        with tf.control_dependencies([]):
            for i, gpu_id in enumerate(args.gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('model_%d' % gpu_id):
                        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                            model = PRCModel(self.vocab, args)
                            model.build_graph(self.input_split[i])
                        model_loss = model.get_loss()
                        tower_predictions.append(model.get_prediction())
                        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        total_loss = tf.add_n([model_loss] + reg_loss)
                        total_losses.append(total_loss)
                        reuse_variables=True
                        grads = self.optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)
        #with tf.control_dependencies(total_losses + total_losses):
        #    self.time1 = self.timestamp()
        self.loss = total_loss
        grads = self.average_gradients(tower_grads, args)
        #grads = tower_grads[0]
        apply_gradients_op = self.optimizer.apply_gradients(grads, self.global_step)
        with tf.control_dependencies([apply_gradients_op]):
            self.train_op = tf.no_op(name='train_op')
            #self.time2 = self.timestamp()
        self.start_probs, self.end_probs = self._gather_output(tower_predictions)
        return self.train_op, self.start_probs, self.end_probs
    
    def _get_label_probs(self, labels, depth, keep_thresh=0.5, fill_value=0):
        sigma = 3
        sigma2 = sigma**2
        f = lambda i, label: np.exp(-(i-label)**2/(2*sigma2)) if label > 0 else fill_value
        label_probs = []
        for label in labels:
            dist = np.array([f(i, label) for i in range(depth)])
            label_probs.append(dist)
        #label_probs = np.concatenate(label_probs, axis=0)
        label_probs = np.vstack(label_probs)
        #print('label_probs shape:{}'.format(label_probs.shape))
        label_probs[label_probs < keep_thresh] = fill_value
        return label_probs

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 1, 0
        t1 = time.time()
        for bitx, batch in enumerate(train_batches, 1):
            t2 = time.time()
            passage_len = len(batch['passage_token_ids'][0])
            label_batch = len(batch['start_id'])
            all_passage = len(batch['passage_token_ids'])
            concat_passage_len = all_passage / label_batch * passage_len
            #print('passage shape:{}'.format(np.array(batch['passage_token_ids']).shape))
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         #self.start_label_probs: self._get_label_probs(batch['start_id'], concat_passage_len),
                         #self.end_label_probs: self._get_label_probs(batch['end_id'], concat_passage_len),
                         self.dropout_keep_prob: dropout_keep_prob}
            _, loss, lr = self.sess.run([self.train_op, self.loss, self.lr], feed_dict)
            t3 = time.time()
            data_t = t2 - t1
            train_t = t3 - t2 
            t1 = t3
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}, lr:{:.6}, data_t:{:.3}, train_t:{:.3}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch, lr, data_t, train_t))
                #self.logger.info('Average loss from batch {} to {} is {}, lr:{:.6}, data_t:{:.3}, train_t:{:.3}, forward_t:{:.4}, average:{:.4}'.format(
                #    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch, lr, data_t, train_t, st1-st0, st2-st1))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            self.logger.info('total batch:{}'.format(data.size('train') // batch_size))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix)

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            passage_len = len(batch['passage_token_ids'][0])
            label_batch = len(batch['start_id'])
            all_passage = len(batch['passage_token_ids'])
            concat_passage_len = all_passage / label_batch * passage_len
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.start_label_probs: self._get_label_probs(batch['start_id'], concat_passage_len),
                         self.end_label_probs: self._get_label_probs(batch['end_id'], concat_passage_len),
                         self.dropout_keep_prob: 1.0}
            if hasattr(self, 'char_vocab'):
                char_input = {
                         self.p_char: batch['passage_char_ids'],
                         self.q_char: batch['question_char_ids'],
                         self.p_char_length: batch['passage_char_length'],
                         self.q_char_length: batch['question_char_length'],
                    }
                feed_dict.update(char_input)
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, encoding='utf8', ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        best_answer = ''.join(
            sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
