#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module implement AP-NN abstract model class.
References:
    `Attentive Pooling Networks`, 2017
"""

import abc

import tensorflow as tf

from model_utils import create_or_load_embed

from functools import reduce


class APModel(object):
    """AP abstract base class."""

    def __init__(self, iterator, params, mode):
        """Initialize model, build graph.
        Args:
          iterator: instance of class BatchedInput, defined in dataset.  
          params: parameters.
          mode: train | eval | predict mode defined with tf.estimator.ModeKeys.
        """
        self.iterator = iterator
        self.params = params
        self.mode = mode
        self.scope = self.__class__.__name__  # instance class name

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            embeddings = create_or_load_embed(
                params.vocab_file, params.embed_file, params.vocab_size, params.embedding_dim)
            self.q = tf.nn.embedding_lookup(embeddings, iterator.q)  # [batch_size, seq_length, embedding_size]
            self.a1 = tf.nn.embedding_lookup(embeddings, iterator.a1)
            # Q, A1, A2 are CNN or biLSTM encoder outputs for question, positive answer, negative answer.
            self.Q = self._encode(self.q, iterator.q_len)  # b * m * c
            self.A1 = self._encode(self.a1, iterator.a1_len)  # b * n * c
            self.r_q, self.r_a1 = self._attentive_pooling(self.Q, self.A1)
            self.score = self._cosine(self.r_q, self.r_a1)
            self._model_stats()  # print model statistics info

            if mode != tf.estimator.ModeKeys.PREDICT:
                self.a2 = tf.nn.embedding_lookup(embeddings, iterator.a2)
                self.A2 = self._encode(self.a2, iterator.a2_len)  # b * n * c
                self.r_q, self.r_a2 = self._attentive_pooling(self.Q, self.A2)
                self.negative_score = self._cosine(self.r_q, self.r_a2)

                with tf.name_scope("loss"):
                    self.loss = tf.reduce_mean(
                        tf.maximum(0.0, self.params.margin - self.score + self.negative_score))

                    if params.optimizer == "rmsprop":
                        opt = tf.train.RMSPropOptimizer(params.lr)
                    elif params.optimizer == "adam":
                        opt = tf.train.AdamOptimizer(params.lr)
                    elif params.optimizer == "sgd":
                        opt = tf.train.MomentumOptimizer(params.lr, 0.9)
                    else:
                        raise ValueError("Unsupported optimizer %s" % params.optimizer)
                    train_vars = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, train_vars)
                    # gradients, _ = opt.compute_gradients(self.loss, train_vars)
                    if params.use_grad_clip:
                        gradients, grad_norm = tf.clip_by_global_norm(
                            gradients, params.grad_clip_norm)

                    self.global_step = tf.Variable(0, trainable=False)
                    self.update = opt.apply_gradients(
                        zip(gradients, train_vars), global_step=self.global_step)

    def _attentive_pooling(self, q, a):
        """Attentive pooling
        Args:
            q: encoder output for question (batch_size, q_len, vector_size)
            a: encoder output for question (batch_size, a_len, vector_size)
        Returns:
            final representation Tensor r_q, r_a for q and a (batch_size, vector_size)
        """
        batch_size = self.params.batch_size
        c = q.get_shape().as_list()[-1]  # vector size
        with tf.variable_scope("attentive-pooling") as scope:
            # G = tanh(Q*U*A^T)  here Q is equal to Q transpose in origin paper.
            self.Q = q  # (b, m, c)
            self.A = a  # (b, n, c)
            self.U = tf.get_variable(
                "U", [c, c],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            self.U_batch = tf.tile(tf.expand_dims(self.U, 0), [batch_size, 1, 1])
            self.G = tf.tanh(
                tf.matmul(
                    tf.matmul(self.Q, self.U_batch), tf.transpose(self.A, [0, 2, 1]))
            )  # G b*m*n

            # column-wise and row-wise max-poolings to generate g_q (b*m*1), g_a (b*1*n)
            g_q = tf.reduce_max(self.G, axis=2, keepdims=True)
            g_a = tf.reduce_max(self.G, axis=1, keepdims=True)

            # create attention vectors sigma_q (b*m), sigma_a (b*n)
            sigma_q = tf.nn.softmax(g_q)
            sigma_a = tf.nn.softmax(g_a)
            # final output r_q, r_a  (b*c)
            r_q = tf.squeeze(tf.matmul(tf.transpose(self.Q, [0, 2, 1]), sigma_q), axis=2)
            r_a = tf.squeeze(tf.matmul(sigma_a, self.A), axis=1)
            return r_q, r_a  # (b, c)

    @staticmethod
    def _cosine(x, y):
        """x, y shape (batch_size, vector_size)"""
        # normalize_x = tf.nn.l2_normalize(x, 0)
        # normalize_y = tf.nn.l2_normalize(y, 0)
        # cosine = tf.reduce_sum(tf.multiply(normalize_x, normalize_y), 1)
        cosine = tf.div(
            tf.reduce_sum(x*y, 1),
            tf.sqrt(tf.reduce_sum(x*x, 1)) * tf.sqrt(tf.reduce_sum(y*y, 1)) + 1e-8,
            name="cosine")
        return cosine

    @abc.abstractmethod
    def _encode(self, x, length):
        """Subclass must implement this method, 
        Returns: 
            An encoder output Tensor, shape: [batch_size, sequence_len, vector_size].
        """
        pass

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))

    def train(self, sess):
            return sess.run([self.update, self.loss])

    def predict(self, sess):
        return sess.run([self.score])

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=self.params.num_keep_ckpts)
        saver.save(sess, path, global_step=self.global_step.eval())

