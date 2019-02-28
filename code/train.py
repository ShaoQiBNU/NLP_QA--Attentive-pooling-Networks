#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
# coding=utf-8
"""This module for model training."""
import os
import time

import tensorflow as tf

from config import FLAGS
from ap_model import AP_CNN, AP_biLSTM
from dataset import get_iterator
from utils import print_args, load_vocab


def train():
    # Training
    tf.set_random_seed(FLAGS.random_seed)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        init_ops = [tf.global_variables_initializer(),
                    tf.local_variables_initializer(), tf.tables_initializer()]
        sess.run(init_ops)

        for epoch in range(FLAGS.max_epoch):
            step = 0
            if FLAGS.use_learning_decay and (epoch+1) % FLAGS.lr_decay_epoch == 0:
                FLAGS.lr *= FLAGS.lr_decay_rate
            print('\nepoch: {}\tlearning rate: {}'.format(epoch+1, FLAGS.lr))

            sess.run(iterator.initializer)
            while True:
                try:
                    _, loss = model.train(sess)
                    step += 1
                    # show train batch metrics
                    if step % FLAGS.stats_per_steps == 0:
                        now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                        # time_str = datetime.datetime.now().isoformat()
                        print('{}\tepoch {:2d}\tstep {:3d}\ttrain loss={:.4f}'.format(
                            now_time, epoch+1, step, loss))
                except tf.errors.OutOfRangeError:
                    print("\n"+"="*25+" Finish train {} epoch ".format(epoch+1)+"="*25+"\n")
                    break

            if (epoch+1) % FLAGS.save_per_epochs == 0:
                if not os.path.exists(FLAGS.model_dir):
                    os.mkdir(FLAGS.model_dir)
                save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                model.save(sess, save_path)
                print("Epoch {}, saved checkpoint to {}".format(epoch+1, save_path))


if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Params Preparation
    print_args(FLAGS)
    vocab_table, _, vocab_size = load_vocab(FLAGS.vocab_file)
    FLAGS.vocab_size = vocab_size

    # Model Preparation
    padding = True if FLAGS.model_type == 1 else False
    mode = tf.estimator.ModeKeys.TRAIN
    iterator = get_iterator(
        FLAGS.train_file, vocab_table, FLAGS.batch_size,
        q_max_len=FLAGS.question_max_len,
        a_max_len=FLAGS.answer_max_len,
        num_buckets=FLAGS.num_buckets,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        padding=padding,
    )
    if FLAGS.model_type == 1:
        model = AP_CNN(iterator, FLAGS, mode)
    else:
        model = AP_biLSTM(iterator, FLAGS, mode)

    train()

