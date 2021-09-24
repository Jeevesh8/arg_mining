import os
from collections import namedtuple
from typing import List, Dict, Tuple, Optional

import tensorflow as tf
import transformers
from transformers import BertTokenizer

from .tokenize_components import get_comp_wise_essays, get_tokenized_essay
from .configs import config as data_config

pe_data = namedtuple("pe_data", ["tokenized_essays", "comp_type_labels",])

def convert_to_named_tuple(tokenized_essays, comp_type_labels):
    return pe_data(tokenized_essays, comp_type_labels)


def data_generator(data_file: str, tokenizer: transformers.PreTrainedTokenizer):
    for essay in get_comp_wise_essays(data_file, tokenizer):
        yield get_tokenized_essay(essay, tokenizer)


def _create_min_max_boundaries(max_length: int,
                               min_boundary: int =256,
                               boundary_scale: float =1.1) -> Tuple[List[int], List[int]]:
    """Forms buckets. All samples with sequence length between the boundaries of a single bucket 
    are to be padded to the same length. 
    Args:
        max_length:     The max_length of any tokenized sequence that will be input to model.
        min_boundary:   The largest sequence length that can be accomodated in the first bucket.
        boundary_scale: The factor by which previous bucket's max length should be multiplied to get
                        the max length for next bucket.
    Returns:
        2 Lists, ``buckets_min`` and ``buckets_max`` the i-th bucket corresponds to sequence lengths 
        from buckets_min[i-1] to buckets_max[i-1].
    """
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset: tf.data.Dataset, batch_size: int, max_length: int, min_boundary: Optional[int]=256) -> tf.data.Dataset:
    """Dynamically batches the samples in dataset. Buckets of similar sample length samples are 
    made. Batches are made from elements in a bucket and each sequence is padded to the max length of
    sample in the batch.
    Args:
        dataset:      A tensorflow dataset to be dynamically batched.
        batch_size:   Number of tokens expected in a single batch. That is the expected sum of 
                      all tokens in all samples in a batch
        max_length:   Max. sequence length any sample from dataset. max_length <= batch_size.
        min_boundary: The max. length of any sequence in the 1-st bucket.
    Returns:
        A tensorflow dataset that returns dynamically batched and padded sample.
    """
    def get_sample_len(*args):
        return tf.shape(args[1])[0]
    
    if batch_size<max_length:
        raise ValueError("The expected number of tokens in a single batch(batch_size) must be \
                         >= max_length of any sequence in dataset. Got batch_size, max_length as:",
                         (batch_size, max_length))
    actual_max_len = 0
    for elem in dataset:
        sample_len = get_sample_len(*elem)
        if actual_max_len<sample_len:
            actual_max_len = sample_len
    
    if actual_max_len > max_length:
        raise AssertionError("Found sequence with longer length than specified: ", max_length,
                             "in dataset. Actual max length:", actual_max_len)
    
    buckets_min, buckets_max = _create_min_max_boundaries(max_length, min_boundary=min_boundary)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(*args) -> int:
        """Returns the index of bucket that the input specified in *args
        falls in."""
        seq_length = get_sample_len(*args)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id: int) -> int:
        """Returns the size of batch for the bucket at index bucket_id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id: int, grouped_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Batches a subset of a dataset, provided in grouped_dataset. Each element is padded 
        to the max. sequence length in the batch. Batches in same bucket may be padded to different 
        lengths.
        """
        bucket_batch_size = window_size_fn(bucket_id)

        return grouped_dataset.padded_batch(bucket_batch_size,
                                            padded_shapes=([None],[None]),
                                            padding_values=tuple(data_config["pad_for"].values()))
    return dataset.apply(
        tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def get_dataset(data_file: List[str], tokenizer: transformers.PreTrainedTokenizer):
    def callable_gen():
        nonlocal data_file
        for elem in data_generator(data_file, tokenizer):
            yield elem

    sample_wise_dataset =  tf.data.Dataset.from_generator(callable_gen,
                                                          output_signature=(
                                                            tf.TensorSpec(shape=(None),
                                                                        dtype=tf.int32,
                                                                        name="tokenized_essays"),
                                                            tf.TensorSpec(shape=(None),
                                                                        dtype=tf.int32,
                                                                        name="comp_type_labels"),
                                                          ))
    
    dataset = _batch_examples(sample_wise_dataset, data_config["batch_size"], data_config["max_len"])
    dataset = dataset.map(convert_to_named_tuple)
    
    return dataset


def load_dataset(
    pe_dir: str = None,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = BertTokenizer.from_pretrained('bert-base-uncased'),
    as_numpy_iter: bool = True,
):
    """Returns a tuple of train, valid, test datasets(according to {train|test|vaild}.txt files in pe_dir)
    Args:
        pe_dir:         The directory having the persuasive essays dataset. Download data from 
                        https://github.com/UKPLab/naacl18-multitask_argument_mining/tree/master/dataSplits/fullData/essays
        
        tokenizer:      The tokenizer to be used for tokenizing the essays. Must inherit from PreTrainedTokenizer.
        
        as_numpy_iter:  Tensorflow dataset is converted to numpy iterator, before returning.

    Returns:
        Tuple of 3 tensorflow datasets, corresponding to train, valid and test data. None is returned for the datasets
        for which no file is present in pe_dir.
    """

    try:
        train_dataset = get_dataset(os.path.join(pe_dir, 'train.txt'), tokenizer)
    except FileNotFoundError:
        train_dataset = None
    
    try:
        valid_dataset = get_dataset(os.path.join(pe_dir, 'dev.txt'))
    except FileNotFoundError:
        valid_dataset = None
    
    try:
        test_dataset = get_dataset(os.path.join(pe_dir, 'test.txt'))
    except FileNotFoundError:
        test_dataset = None
    

    if as_numpy_iter:
        train_dataset = (None if train_dataset is None else
                         train_dataset.as_numpy_iterator())
        valid_dataset = (None if valid_dataset is None else
                         valid_dataset.as_numpy_iterator())
        test_dataset = (None if test_dataset is None else
                        test_dataset.as_numpy_iterator())

    return train_dataset, valid_dataset, test_dataset

def get_pad_mask(batch, dtype=tf.float32):
    """Returns a mask for the tokenized threads in batch 
    with 0 where there is pad token, 1 elsewhere."""
    return tf.cast(batch!=data_config["pad_for"]["tokenized_essays"], dtype=dtype)

def get_user_tokens_mask(batch, dtype=tf.float32):
    """Returns a mask for the tokenized_threads in batch
    with 1 where a user token's id is there, 0 elsewhere."""
    user_tokens_mask = tf.zeros_like(batch[:,:-1])
    user_tokens_mask = tf.pad(user_tokens_mask, paddings=[[0,0], [1,0]], constant_values=1)
    return tf.cast(user_tokens_mask, dtype=dtype)
