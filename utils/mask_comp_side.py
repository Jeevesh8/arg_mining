import random
from typing import Tuple, List, Generator

import numpy as np

from arg_mining.datasets.cmv_modes import data_config

def remove_comps(anns: List[List[str]], masked_threads: List[List[int]], mask_token_id: int, dm_range: int, with_dms: bool):
    """Removes components annotated near masked tokens in masked_threads, if with_dms=True,
    else removes components not near masked tokens in masked_threads."""
    
    def get_refined_comp(ann, masked_thread, start_idx, end_idx):
        if (mask_token_id in masked_thread[start_idx-dm_range:start_idx+dm_range] or
            mask_token_id in masked_thread[end_idx-dm_range:end_idx+dm_range]):
            if with_dms:
                return ann[start_idx:end_idx]
            else:
                return ["O"]*(end_idx-start_idx)
        else:
            if with_dms:
                return ["O"]*(end_idx-start_idx)
            else:
                return ann[start_idx:end_idx]
    
    new_anns = []
    for ann, masked_thread in zip(anns, masked_threads):
        assert len(ann) == len(masked_thread)
        new_ann = []
        i=0
        while i<len(ann):
            if ann[i]=="B-C":
                start_idx = i
                i += 1
                while i<len(ann) and ann[i]=="I-C":
                    i += 1
                end_idx = i
                new_ann.extend(get_refined_comp(ann, masked_thread, start_idx, end_idx))
            elif ann[i]=="B-P":
                start_idx = i
                i+=1
                while i<len(ann) and ann[i]=="I-P":
                    i += 1
                end_idx = i
                new_ann.extend(get_refined_comp(ann, masked_thread, start_idx, end_idx))
            else:
                new_ann.append(ann[i])
                i += 1
        new_anns.append(new_ann)
    
    return new_anns
        
def get_masked_data_lists(dataset, 
                          tokenizer,
                          left: bool=True) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Args:
        dataset:    A python generator that yields tuples of np.array's
                    consisting of tokenized_threads, masked_threads and component
                    type labels
        left:       If true, left side of components are masked. Otherwise right
                    side is masked.
    Returns:
        A tuple of two lists consisting of samples from entire dataset:
            final_threads:  A list of lists of int. Where each internal list corresponds
                            to a thread masked on one side.
            final_labels:   A list of lists of int. Where each internal list corresponds
                            to a masked component type labels.
    NOTE:
        A left masked sample consists of a tokenized thread whose all tokens before
        the beginning of some argumentative component are [MASK] and the corresponding
        component type labels are "other".
    """
    final_threads, final_labels = [], []
    if not left:
        for (tokenized_threads, _masked_threads, comp_type_labels) in dataset:
            for (tokenized_thread, comp_type_label) in zip(tokenized_threads, comp_type_labels):
                tokenized_thread = tokenized_thread.tolist()
                comp_type_label = comp_type_label.tolist()
                left_masked_thread = []
                comp_types_for_left_masked_thread = []
                for i, (_token, label) in enumerate(zip(tokenized_thread, comp_type_label)):
                    if (label == data_config["arg_components"]["B-C"] or 
                        label == data_config["arg_components"]["B-P"]):
                        final_threads.append(left_masked_thread+tokenized_thread[i:])
                        final_labels.append(comp_types_for_left_masked_thread+comp_type_label[i:])
                    left_masked_thread.append(tokenizer.mask_token_id)
                    comp_types_for_left_masked_thread.append(0)
    else:
        for (tokenized_threads, _masked_threads, comp_type_labels) in dataset:
            for (tokenized_thread, comp_type_label) in zip(tokenized_threads, comp_type_labels):
                tokenized_thread = tokenized_threads.tolist()[::-1]
                comp_type_label = comp_type_label.tolist()[::-1]
                right_masked_thread = []
                comp_types_for_right_masked_thread = []
                for i, (_token, label) in enumerate(zip(tokenized_thread, comp_type_label)):
                    if (label == data_config["arg_components"]["I-C"] or 
                        label == data_config["arg_components"]["I-P"]):
                        final_threads.append(right_masked_thread+tokenized_thread[i:])
                        final_labels.append(comp_types_for_right_masked_thread+comp_type_label[i:])
                        final_threads[-1] = final_threads[-1][::-1]
                        final_labels[-1] = final_labels[-1][::-1]
                    right_masked_thread.append(tokenizer.mask_token_id)
                    comp_types_for_right_masked_thread.append(0)
    return final_threads, final_labels

def get_masked_dataset(dataset,
                       tokenizer,
                       left:bool = True,
                       shuffle:bool = True,
                       batch_size: int = 4) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Args:
        dataset:    Same as in get_masked_data_lists()
        left:       Same as in get_masked_data_lists()
        shuffle:    Whether to shuffle around the elements corresponding to masking
                    of various threads. If True, batch will consist of random left/right
                    masked samples from different tokenized_threads, rather than same one.
        batch_size: Number of elements to put in a batch. 
    
    Yields:
        A batch consisting of a tuple of np.array's corresponding to left/right masked
        tokenized_threads, and comp_type_labels.
    """
    masked_threads, labels_for_masked_threads = get_masked_data_lists(dataset, left)
    samples =[(elem1, elem2) for (elem1, elem2) in zip(masked_threads, labels_for_masked_threads)]
    if shuffle:
        random.shuffle(samples)
    
    batch_threads = []
    batch_labels = []
    lengths = []
    for sample in samples:
        batch_threads.append(sample[0])
        batch_labels.append(sample[1])
        lengths.append(len(sample[0]))
        if len(batch_threads)==batch_size:
            max_len = max(lengths)
            for thread, label in zip(batch_threads, batch_labels):
                thread += [tokenizer.pad_token_id]*(max_len-len(thread))
                label += [data_config["pad_for"]["comp_type_labels"]]*(max_len-len(thread))
            yield np.array(batch_threads), np.array(batch_labels)
            batch_threads, batch_labels, lengths = [], [], []
