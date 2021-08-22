import os, shlex, glob, subprocess
import random
from typing import List, Dict, Optional
from collections import namedtuple
from functools import partial

import tensorflow as tf
import transformers
from transformers import BertTokenizer
from bs4 import BeautifulSoup

from .tokenize_components import get_model_inputs
from .configs import config as data_config

cmv_modes_data = namedtuple(
    "cmv_modes_data",
    [
        "filenames", "tokenized_threads", "masked_thread", "comp_type_labels",
        "refers_to_and_type"
    ],
)

if data_config["omit_filenames"]:
    cmv_modes_data = namedtuple(
        "cmv_modes_data",
        ["tokenized_threads", "masked_thread", "comp_type_labels", "refers_to_and_type"],
    )


def convert_to_named_tuple(filenames, tokenized_threads, masked_thread, 
                           comp_type_labels, refers_to_and_type, omit_filenames):
    if omit_filenames:
        return cmv_modes_data(tokenized_threads, masked_thread, comp_type_labels,
                              refers_to_and_type)
    return cmv_modes_data(filenames, tokenized_threads, masked_thread, comp_type_labels,
                          refers_to_and_type)


convert_to_named_tuple = partial(convert_to_named_tuple,
                                 omit_filenames=data_config["omit_filenames"])


def data_generator(file_list: List[str], tokenizer: transformers.PreTrainedTokenizer, mask_tokens: Optional[List[str]] = None):
    for elem in get_model_inputs(file_list, tokenizer, mask_tokens):
        yield elem


def get_dataset(file_list: List[str], tokenizer: transformers.PreTrainedTokenizer, mask_tokens: Optional[List[str]] = None):
    
    if data_config["pad_for"]["tokenized_thread"] is None:
        data_config["pad_for"]["tokenized_thread"] = tokenizer.pad_token_id
    
    def callable_gen():
        nonlocal file_list
        for elem in data_generator(file_list, tokenizer, mask_tokens):
            yield elem

    return (tf.data.Dataset.from_generator(
        callable_gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string, name="filenames"),
            tf.TensorSpec(shape=(None),
                          dtype=tf.int32,
                          name="tokenized_threads"),
            tf.TensorSpec(shape=(None),
                          dtype=tf.int32,
                          name="masked_threads"),
            tf.TensorSpec(shape=(None),
                          dtype=tf.int32,
                          name="comp_type_labels"),
            tf.TensorSpec(shape=(None, 3),
                          dtype=tf.int32,
                          name="refers_to_and_type"),
        ),
    ).padded_batch(
        data_config["batch_size"],
        padded_shapes=(
            [],
            [data_config["max_len"]],
            [data_config["max_len"]],
            [data_config["max_len"]],
            [data_config["max_len"], 3],
        ),
        padding_values=(None, data_config["pad_for"]["tokenized_thread"], 
                              data_config["pad_for"]["tokenized_thread"], 
                              data_config["pad_for"]["comp_type_labels"], 
                              data_config["pad_for"]["refers_to_and_type"]),
    
    ).batch(data_config["num_devices"],
            drop_remainder=True).map(convert_to_named_tuple))


def get_op_wise_split(filelist: List[str]) -> Dict[str, List[str]]:
    """Splits up filelist into groups, each of which corresponds to threads with
    the same original posts.
    """
    splits = {}
    for filepath in filelist:
        if not filepath.endswith(".xml"):
            continue
        with open(filepath) as f:
            xml_string = f.read()
        parsed_xml = BeautifulSoup(xml_string, "lxml")
        op_id = parsed_xml.thread["id"]
        if op_id not in splits:
            splits[op_id] = []
        splits[op_id].append(filepath)
    return splits


def load_dataset(
    cmv_modes_dir: str = None,
    mask_tokens: Optional[List[str]] = None,
    tokenizer : Optional[transformers.PreTrainedTokenizer] = BertTokenizer.from_pretrained('bert-base-uncased'),
    train_sz: float = 100,
    valid_sz: float = 0,
    test_sz: float = 0,
    shuffle: bool = False,
    as_numpy_iter: bool = True,
):
    """Returns a tuple of train, valid, test datasets(the ones having non-zero size only)
    Args:
        cmv_modes_dir:  The directory to the version of cmv modes data from which the dataset is to be loaded.
                        If None, the data is downloaded into current working directory and v2.0 is used from there.
        mask_tokens:    A list of strings to be masked from each thread. The masking is done in the de-tokenized string. 
                        Any instance of any string in this list in the thread will be replaced by apt number of <mask> tokens.
        tokenizer:      The tokenizer to use, by default uses 'bert-base-uncased'. The provided tokenizer must inherit from 
                        transformers.PreTrainedTokenizer.
        train_sz:       The % of total threads to include in train data. By default, all the threads are included in train_data.
        valid_sz:       The % of total threads to include in validation data.
        test_sz:        The % of total threads to include in testing data.
        shuffle:        If True, the data is shuffled before splitting into train, test and valid sets.
        as_numpy_iter:  Tensorflow dataset is converted to numpy iterator, before returning.
    Returns:
        Tuple of 3 tensorflow datasets, corresponding to train, valid and test data. None is returned for the datasets
        for which size of 0 was specified.
    """
    if train_sz + valid_sz + test_sz != 100:
        raise ValueError("Train, test valid sizes must sum to 100")

    if cmv_modes_dir is None:
        subprocess.call(shlex.split("git clone https://github.com/chridey/change-my-view-modes"))
        cmv_modes_dir = os.path.join(os.getcwd(), "change-my-view-modes/v2.0/")

    splits = get_op_wise_split(glob.glob(os.path.join(cmv_modes_dir, "*/*")))

    op_wise_splits_lis = list(splits.values())
    if shuffle:
        random.shuffle(op_wise_splits_lis)

    n_threads = sum([len(threads) for threads in splits.values()])
    num_threads_added = 0
    train_files, valid_files, test_files, i = [], [], [], 0

    if train_sz != 0:
        while (num_threads_added /
               n_threads) * 100 < train_sz and i < len(op_wise_splits_lis):
            train_files += op_wise_splits_lis[i]
            num_threads_added += len(op_wise_splits_lis[i])
            i += 1
        num_threads_added = 0

    if valid_sz != 0:
        while (num_threads_added /
               n_threads) * 100 < valid_sz and i < len(op_wise_splits_lis):
            valid_files += op_wise_splits_lis[i]
            num_threads_added += len(op_wise_splits_lis[i])
            i += 1
        num_threads_added = 0

    if test_sz != 0:
        while (num_threads_added /
               n_threads) * 100 <= test_sz and i < len(op_wise_splits_lis):
            test_files += op_wise_splits_lis[i]
            num_threads_added += len(op_wise_splits_lis[i])
            i += 1

    train_dataset = None if len(train_files) == 0 else get_dataset(train_files, tokenizer, mask_tokens)
    valid_dataset = None if len(valid_files) == 0 else get_dataset(valid_files, tokenizer, mask_tokens)
    test_dataset = None if len(test_files) == 0 else get_dataset(test_files, tokenizer, mask_tokens)

    if as_numpy_iter:
        train_dataset = (None if train_dataset is None else
                         train_dataset.as_numpy_iterator())
        valid_dataset = (None if valid_dataset is None else
                         valid_dataset.as_numpy_iterator())
        test_dataset = (None if test_dataset is None else
                        test_dataset.as_numpy_iterator())

    return train_dataset, valid_dataset, test_dataset
