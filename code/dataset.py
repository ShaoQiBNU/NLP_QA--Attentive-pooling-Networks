#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module contains efficient data read and transform using tf.data API.
Train data format:
    question \t positive answer \t negative answer
Prediction data format:
    question \t positive answer
"""
import collections

import tensorflow as tf


class BatchedInput(
    collections.namedtuple(
        "BatchedInput", ("initializer", "q", "a1", "a2",
                         "q_len", "a1_len", "a2_len"))):
    """
    q for question; a1 for positive answer; a2 for negtive answer; 
    q_len, a1_len, a2_len for each sequence length.
    for inference, a1 for answer, set a2, a2_len None.
    """
    pass


def _parse_csv(line):
    cols_types = [['']] * 3
    columns = tf.decode_csv(line, record_defaults=cols_types, field_delim='\t',  use_quote_delim=False)
    return columns


def _parse_infer_csv(line):
    cols_types = [['']] * 2
    columns = tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')
    return columns


def get_infer_iterator(data_file, vocab_table, batch_size, q_max_len=None, a_max_len=None, padding=False):
    """Iterator for inference.
    Args:
        data_file: data file, each line contains question, answer
        vocab_table: tf look-up table
        q_max_len: question max length
        a_max_len: answer max length
        padding: Bool
            set True for cnn model to pad all samples into same length, must set seq_max_len
            set False for rnn model 
    Returns:
        BatchedInput instance
            (initializer, question ids, answer ids, question length, answer length).
    """
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(_parse_infer_csv).prefetch(batch_size)
    dataset = dataset.map(
        lambda q, a: (tf.string_split([q]).values, tf.string_split([a]).values))
    if q_max_len:
        dataset = dataset.map(lambda q, a: (q[:q_max_len], a))
    if a_max_len:
        dataset = dataset.map(lambda q, a: (q, a[:a_max_len]))
    # Convert the word strings to ids
    dataset = dataset.map(
        lambda q, a: (tf.cast(vocab_table.lookup(q), tf.int32),
                      tf.cast(vocab_table.lookup(a), tf.int32),
                      tf.size(q),
                      tf.size(a)))

    question_pad_size = q_max_len if padding else None
    answer_pad_size = a_max_len if padding else None
    batched_dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([question_pad_size]),
            tf.TensorShape([answer_pad_size]),
            tf.TensorShape([]), tf.TensorShape([])),
        padding_values=(0, 0, 0, 0))

    batched_iter = batched_dataset.make_initializable_iterator()
    q_ids, a_ids, q_len, a_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        q=q_ids, a1=a_ids, a2=None, q_len=q_len, a1_len=a_len, a2_len=None)


def get_iterator(data_file,
                 vocab_table,
                 batch_size,
                 num_buckets=1,
                 q_max_len=None,
                 a_max_len=None,
                 padding=False,
                 num_parallel_calls=4,
                 shuffle_buffer_size=None):
    """Iterator for train and eval.
    Args:
        data_file: data file, each line contains question, answer_pos, answer_neg 
        vocab_table: tf look-up table
        q_max_len: question max length
        a_max_len: answer max length, both positive and negative
        padding: Bool
            set True for cnn or attention based model to pad all samples into same length, must set seq_max_len
            set False for rnn model 
        num_buckets: bucket according to sequence length
        shuffle_buffer_size: buffer size for shuffle
    Returns:
        BatchedInput instance
            (initializer, question ids, pos answer ids, neg answer ids, question length, pos answer length, neg answer length).
    """
    shuffle_buffer_size = shuffle_buffer_size or batch_size * 1000

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.map(_parse_csv)
    dataset = dataset.map(
        lambda q, a1, a2: (tf.string_split([q]).values, tf.string_split([a1]).values, tf.string_split([a2]).values),
        num_parallel_calls=num_parallel_calls)
    if q_max_len:
        dataset = dataset.map(
            lambda q, a1, a2: (q[:q_max_len], a1, a2),
            num_parallel_calls=num_parallel_calls)
    if a_max_len:
        dataset = dataset.map(
            lambda q, a1, a2: (q, a1[:a_max_len], a2[:a_max_len]),
            num_parallel_calls=num_parallel_calls)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(
        lambda q, a1, a2: (tf.cast(vocab_table.lookup(q), tf.int32),
                           tf.cast(vocab_table.lookup(a1), tf.int32),
                           tf.cast(vocab_table.lookup(a2), tf.int32),
                           tf.size(q), tf.size(a1), tf.size(a2)),
        num_parallel_calls=num_parallel_calls)

    question_pad_size = q_max_len if padding else None
    answer_pad_size = a_max_len if padding else None

    if num_buckets > 1:  # Bucket by sequence length (buckets for lengths 0-9, 10-19, ...)
        buckets_length = a_max_len // num_buckets
        buckets_boundaries = [buckets_length * (i+1) for i in range(num_buckets)]
        buckets_batch_sizes = [batch_size] * (len(buckets_boundaries) + 1)

        batching_func = tf.contrib.data.bucket_by_sequence_length(
            element_length_func=lambda q, a1, a2, q_len, a1_len, a2_len: (a1_len + a2_len) // 2,
            bucket_boundaries=buckets_boundaries,
            bucket_batch_sizes=buckets_batch_sizes,
            padded_shapes=(
                tf.TensorShape([question_pad_size]),
                tf.TensorShape([answer_pad_size]),
                tf.TensorShape([answer_pad_size]),
                tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
            padding_values=(0, 0, 0, 0, 0, 0),
            pad_to_bucket_boundary=False
        )
        batched_dataset = dataset.apply(batching_func).prefetch(2*batch_size)
    else:
        batching_func = tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size,
            padded_shapes=(
                tf.TensorShape([question_pad_size]),
                tf.TensorShape([answer_pad_size]),
                tf.TensorShape([answer_pad_size]),
                tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0, 0, 0, 0, 0, 0)
        )
        batched_dataset = dataset.apply(batching_func).prefetch(2 * batch_size)

        # Note tf.data default to include last smaller batch, it cause error.
        # From tf version >= 1.10, we can use drop_remainder options
        # batched_dataset = dataset.padded_batch(
        #     batch_size,
        #     padded_shapes=(
        #         tf.TensorShape([question_pad_size]),
        #         tf.TensorShape([answer_pad_size]),
        #         tf.TensorShape([answer_pad_size]),
        #         tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
        #     padding_values=(0, 0, 0, 0, 0, 0),
        #     drop_remainder=True
        # ).prefetch(2*batch_size)

    batched_iter = batched_dataset.make_initializable_iterator()
    q_ids, a1_ids, a2_ids, q_len, a1_len, a2_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        q=q_ids, a1=a1_ids, a2=a2_ids, q_len=q_len, a1_len=a1_len, a2_len=a2_len)

