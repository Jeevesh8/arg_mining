import os, warnings
import tensorflow as tf
from typing import Any, List, Dict, Tuple, Optional, Union

import transformers

from .configs import config
from .component_generator import generate_components
from .utils import convert_outputs_to_tensors, get_rel_type_idx, reencode_mask_tokens


def get_arg_comp_lis(comp_type: str, length: int) -> List[str]:
    """Returns a list of labels for a component of comp_type of specified length."""
    if comp_type not in ["claim", "premise"]:
        raise ValueError("Un-supported component type: " + comp_type +
                         " Try changing 'arg_components' in config")
    comp_type = "C" if comp_type == "claim" else "P"
    begin = config["arg_components"]["B-" + comp_type]
    intermediate = config["arg_components"]["I-" + comp_type]
    return [begin] + [intermediate] * (length - 1)


def get_ref_link_lis(related_to: int, first_idx: int,
                     last_idx: int) -> List[int]:
    """To be used for getting token ids to link the component between [first_index, last_index) to the component at distance related_to.
    Every token except the first, refers to its previous token.
    Args:
        related_to: The distance of the related component from(beginning of previous comment or any other place).
        first_idx:  The first index of the component whose tokens need to be linked.
    Returns: a List of the token positions that each token of the component beginning at first_idx refers to.
    """
    try:
        refs = [config["dist_to_label"][related_to]]
    except KeyError:
        refs = [0]
    return refs + [
        config["dist_to_label"][i] for i in range(first_idx, last_idx - 1)
    ]


def get_global_attention(tokenized_thread: List[int],
                         user_token_indices: List[int]) -> List[int]:
    global_attention = [0] * len(tokenized_thread)
    for i, elem in enumerate(tokenized_thread):
        if elem in user_token_indices:
            global_attention[i] = 1
    return global_attention


def find_last_to_last(lis, elem_set) -> int:
    """Returns the index of last to last occurance of any element of elem_set in lis,
    if element is not found at least twice, returns -1."""
    count = 0
    for idx, elem in reversed(list(enumerate(lis))):
        if elem in elem_set:
            count += 1
        if count == 2:
            return idx
    return 0


def get_tokenized_thread(
    filename: str,
    tokenizer: transformers.PreTrainedTokenizer,
    mask_tokens: Optional[List[str]]=None
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, Tuple[
        str, str]], Dict[str, int], Dict[str, str], ]:
    """Returns the tokenized version of thread present in filename. File must be an xml file of cmv-modes data.
    Returns a tuple having:
        tokenized_thread: A 1-D integer List containing the input_ids output by tokenizer for the text in filename.
        begin_positions:  A dictionary mapping the ids of various argumentative components to their beginning index in tokenized_thread.
        ref_n_rel_type:  A dictionary mapping the ids of various argumentative components to a tuple of the form
                        ('_' separated string of all components the component relates to, relation type)
        end_positions: A dictionary mapping the ids of various argumentative components to their ending index+1 in tokenized_thread.
        comp_types: A dictionary mapping the ids of various argumentative components to their types ('claim'/'premise')
    """
    def adjust_ref_rel_type(ref_n_rel_type, begin_positions):
        """Removes any links in ref_n_rel_type which point to some component outside of those in
        the keys of begin_positions."""
        for comp_id, (refers, rel_type) in ref_n_rel_type.items():
            final_refs = [
                ref for ref in str(refers).split("_") if ref in begin_positions
            ]
            ref_n_rel_type[comp_id] = ((None, None) if refers is None else
                                       ("_".join(final_refs), rel_type))

    begin_positions = {}
    end_positions = {}
    ref_n_rel_type = {}
    comp_types = {}

    masked_thread = [tokenizer.bos_token_id]
    tokenized_thread = [tokenizer.bos_token_id]
    
    for component_tup in generate_components(filename):
        component, comp_type, comp_id, refers, rel_type = component_tup
        encoding = tokenizer.encode(component)[1:-1]
        if len(encoding) == 0:
            print("Empty component: ", encoding, component)

        if len(encoding) + len(tokenized_thread) >= config["max_len"] - 1:
            adjust_ref_rel_type(ref_n_rel_type, begin_positions)
            break

        if mask_tokens is not None:
            encoding, masked_encoding = reencode_mask_tokens(encoding, tokenizer,
                                                             mask_tokens)
            encoding, masked_encoding = encoding[1:-1], masked_encoding[1:-1]
            masked_thread += masked_encoding
        else:
            masked_thread += encoding

        if comp_type in ["claim", "premise"]:
            begin_positions[comp_id] = len(tokenized_thread)
            end_positions[comp_id] = len(tokenized_thread) + len(encoding)
            ref_n_rel_type[comp_id] = (refers, rel_type)
            comp_types[comp_id] = comp_type
        
        tokenized_thread += encoding

        if len(begin_positions) >= config["max_comps"] - 1:
            adjust_ref_rel_type(ref_n_rel_type, begin_positions)
            break

    tokenized_thread.append(tokenizer.eos_token_id)
    masked_thread.append(tokenizer.eos_token_id)

    return (
        tokenized_thread,
        masked_thread,
        begin_positions,
        ref_n_rel_type,
        end_positions,
        comp_types,
    )


def check_component_indices(prev_end: int, begin: int, end: int,
                            tokenized_thread: List[Any]):
    """Raises Error if two adjacent components are overlapping, or the boundaries
    of if the components lie outside the boundaries of tokenized_thread.
    Args:
        prev_end:           The index where the previous component ends in tokenized_thread
        begin:              The index where the component to be checked begins in tokenized_thread
        end:                The index where the component to be checked ends in tokenized_thread
        tokenized_thread:   The thread containing all the components.
    Returns:
        None
    """
    if prev_end > begin:
        raise AssertionError(
            "Overlapping components! End of previous component: " +
            str(prev_end) + " .Beginning of next component: " + str(begin))

    if not (0 <= begin <= end < len(tokenized_thread)
            or 0 <= prev_end + 1 <= begin <= end < len(tokenized_thread)):
        raise AssertionError("Begin, and end are not correct." + str(begin) +
                             ", " + str(end))


@convert_outputs_to_tensors(dtype=tf.int32)
def get_thread_with_labels(
    filename,
    tokenizer: transformers.PreTrainedTokenizer,
    mask_tokens: Optional[List[str]] = None
) -> Tuple[List[int], List[int], List[List[int]]]:
    """Returns the tokenized threads along with all the proper labels.
    Args:
        filename:       The xml whose data is to be tokenized.
        tokenizer:      The tokenizer to use to tokenize the cmv data. Must inherit from PreTrainedTokenizer.
        mask_tokens:    A list of tokens to mask.
    Returns a tuple having:
        tokenized_thread:     A 1-D integer List containing the input_ids output by tokenizer for the text in filename.
        masked_thread:        A 1-D integer List with the ids corresponding to the tokens provided in mask_tokens masked from the tokenized_thread.
        comp_type_labels:     A 1-D integer List containing labels for the type of component(other, begin-claim, inter-claim..) [size = len(tokenized_thread)]
        refers_to_and_type:   A 2-D integer List where a single entry [i,j,k] corresponds to a link of type k from i-th component to j-th component.
                              Component numbers are 1-indexed. A link to component 0, means linked to no component. Single component can be linked to multiple
                              components, but only the first one is included in this.
    """
    (
        tokenized_thread,
        masked_thread,
        begin_positions,
        ref_n_rel_type,
        end_positions,
        comp_types,
    ) = get_tokenized_thread(filename, tokenizer, mask_tokens)

    comp_type_labels = [config["arg_components"]["other"]
                        ] * len(tokenized_thread)

    prev_end = 0
    refer_to_and_type = []

    begin_pos_lis = [v for _, v in begin_positions.items()]
    begin_pos_lis.sort()

    for comp_id in begin_positions:
        ref, rel = ref_n_rel_type[comp_id]
        begin, end = begin_positions[comp_id], end_positions[comp_id]

        check_component_indices(prev_end, begin, end, tokenized_thread)

        comp_type_labels[begin:end] = get_arg_comp_lis(comp_types[comp_id],
                                                       end - begin)

        rel_type = get_rel_type_idx(str(rel))

        for j, ref_id in enumerate(str(ref).split("_")):
            if j == 0:
                if ref_id == "None":

                    refer_to_and_type.append(
                        (begin_pos_lis.index(begin) + 1, 0, rel_type))

                elif comp_id != ref_id:

                    refer_to_and_type.append((
                        begin_pos_lis.index(begin) + 1,
                        begin_pos_lis.index(begin_positions[ref_id]) + 1,
                        rel_type,
                    ))
                else:
                    warnings.warn("Skipping Link from component: " + comp_id +
                                  " to itself detected in: " + filename)
            else:
                warnings.warn(
                    "Skipping the extra link for component: " + comp_id +
                    " to " + ref_id + " for file: " + filename, )

        prev_end = end

    if len(tokenized_thread) != len(comp_type_labels):
        raise AssertionError("Incorrect Dataset Loading !!")

    return (
        tokenized_thread,
        masked_thread,
        comp_type_labels,
        refer_to_and_type,
    )


def get_model_inputs(file_lis: Union[List[str], str], tokenizer: transformers.PreTrainedTokenizer, mask_tokens: Optional[List[str]]=None):
    if type(file_lis) is str:
        if not os.path.isdir(file_lis):
            raise ValueError(
                "get_model_inputs() take either a directory name or file list as input! The provided argument is incorrect."
            )
        file_lis = [os.path.join(file_lis, f) for f in os.listdir(file_lis)]
    for filename in file_lis:
        if not (os.path.isfile(filename) and filename.endswith(".xml")):
            continue
        yield (tf.constant(filename), *get_thread_with_labels(filename, tokenizer, mask_tokens))
