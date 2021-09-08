import copy
from functools import wraps
from typing import List, Optional, Tuple

import tensorflow as tf
from .configs import config


def convert_outputs_to_tensors(dtype):
    def inner(func):
        @wraps(func)
        def tf_func(*args, **kwargs):
            outputs = func(*args, **kwargs)
            return tuple(
                (tf.convert_to_tensor(elem, dtype=dtype) for elem in outputs))

        return tf_func

    return inner


def find_sub_list(sl, l):
    """
    Returns the start and end positions of sublist sl in l
    """
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll


def is_internal(enclosing_string: str,
                begin: int,
                end: int,
                prev_str: Optional[str] = None) -> bool:
    """Returns whether the word from begin index to end index in enclosing_string is
    internal. Optionally takes a prev_str which is suuposed to be any string that occurs immediately
    to the left of enclosing string. It is assumed nothing to the right of enclosing_string.

    For e.g. :
        The substring "so" in "I love Alphonso Mangos!" is internal.
        The substring "so" in "And,so I went to their house.." or "And, so I went to their house.." is external.
        The substring "So" in "So, what was I supposed to do?" is external.
        Both "so" are considered external in "It was so-so."
    """
    return ((begin > 0 and enclosing_string[begin - 1].isalnum()) or
            (end < len(enclosing_string) and enclosing_string[end].isalnum())
            or (begin == 0 and
                (True if prev_str is None else prev_str[-1].isalnum())
                and enclosing_string[0].isalnum()))


def has_space(
    enclosing_string: str,
    idx: int,
    preceeding: bool = True,
    prev_str: Optional[str] = None,
) -> bool:
    """Returns whether there is a space preceeding/following(based on whether preceeding is True or False respectively)
    idx in the enclosing_string, which may optionally have a prev_str concatenated to its immediate left.
    It is assumed nothing to the right of enclosing_string.
    """
    if preceeding:
        return (idx == 0 and (True if prev_str is None else prev_str[-1]
                              == " ")) or enclosing_string[idx - 1] == " "

    return idx == len(enclosing_string) - 1 or enclosing_string[idx + 1] == " "


def modified_mask_encodings(
        encoding: List[int], tok,
        markers: List[str]) -> Tuple[List[int], List[List[int]]]:
    """Modifies the text corresponding to encoding, to add spaces before discourse markers and finds the apt encodings
    to use for the matches of markers in the resulting text.

    Args:
        encoding: Encoding of original text in the data.
        tok:      The tokenizer being used. Must implement encode(), decode() methods that add <s>, </s> tokens
        markers:  List of markers to be masked. Markers will be masked in a case-insensistive way, and the case
                 of the original text will be preserved.
    Returns:
        A tuple whose first element is encoding of the modified text(with spaces before and after each occurance of markers in original text,
        whose encoding was provided as input) and second element is a list of sequences corresponding to the sequences to be masked in the
        encoding of modified text. A sequence is repeated as mnay times as it occurs in encoding.

    Note:
        A marker is said to be "matched" to some part of resulting text iff both the part and the marker consist of same
        characters(may have different cases), and the immediate characters to the left and right of the part are either
        string boundaries or non-alphanumeric characters.
    """
    markers = [marker.strip() for marker in markers]

    if encoding[0] != tok.bos_token_id:
        encoding = [tok.bos_token_id] + encoding
    if encoding[-1] != tok.eos_token_id:
        encoding = encoding + [tok.eos_token_id]
    
    sos_length = len(tokenizer.convert_ids_to_tokens([tokenizer.bos_token_id])[0])
    eos_length = len(tokenizer.convert_ids_to_tokens([tokenizer.eos_token_id])[0])
    decoded_txt = tok.decode(encoding)[sos_length:-eos_length]
    special_tokens = tok.get_added_vocab().keys()
    new_txt_parts = []
    iter_txt = decoded_txt
    seqs_to_mask = []
    while True:
        found_markers_start_pos = [
            iter_txt.lower().find(marker.lower()) for marker in markers
        ]
        idx = min(filter(lambda x: x >= 0, found_markers_start_pos),
                  default=-1)
        marker = markers[found_markers_start_pos.index(idx)]
        if idx == -1:
            new_txt_parts.append(iter_txt)
            break

        prev_part = None if len(new_txt_parts) == 0 else new_txt_parts[-1]

        if not is_internal(
                iter_txt, idx, idx + len(marker), prev_str=prev_part):

            # Add the text around detected marker to new_txt_parts, in a format that
            # matches(when encoded) to the mask made for the detected marker in following step.
            new_txt_parts.append(iter_txt[:idx])

            if not has_space(iter_txt, idx, prev_str=prev_part):
                new_txt_parts.append(" ")

            new_txt_parts.append(iter_txt[idx:idx + len(marker)])

            if not has_space(iter_txt, idx + len(marker) -1, preceeding=False):
                new_txt_parts.append(" ")

            # Form the sequence to mask for the detected marker.
            words_before = iter_txt[:idx].strip().split()
            if idx == 0 or (len(words_before) > 0
                            and words_before[-1] in special_tokens):  # Note 1

                to_mask = tok.encode(iter_txt[idx:idx + len(marker)])[1:-1]
            else:
                to_mask = tok.encode(" " +
                                     iter_txt[idx:idx + len(marker)])[1:-1]
            seqs_to_mask.append(to_mask)

        else:
            new_txt_parts.append(iter_txt[:idx + len(marker)])

        iter_txt = iter_txt[idx + len(marker):]

    new_encoding = tok.encode("".join(new_txt_parts))

    return new_encoding, seqs_to_mask


def reencode_mask_tokens(encoding: List[int], tok,
                         markers: List[str]) -> Tuple[List[int], List[int]]:
    """
    Args:
        Same as modified_mask_encodings()
    Returns:
        The modified encoding for optimal matching of markers in the text corresponding to encoding,
        and the encoding with all the matches masked.
    """
    new_encoding, seqs_to_mask = modified_mask_encodings(
        encoding, tok, markers)
    masked_encoding = copy.deepcopy(new_encoding)
    mask_token = tok.mask_token_id
    for seq in seqs_to_mask:
        start, end = find_sub_list(seq, masked_encoding)
        masked_encoding[start:end] = [mask_token] * (end - start)
    return new_encoding, masked_encoding


def get_rel_type_idx(relation: str) -> int:
    for i, v in enumerate(config["relations_map"].values()):
        if relation in v:
            return i
    return 0  # Assuming None relation is 0-th position, always.


"""
NOTE 1: Since, tokenization is same for "[STARTQ] On the.." and "[STARTQ]On the..", 
        if the detected marker is following a special token, no space preceeding it is added to form the sequence to mask.
"""
