import html, re
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Optional

import transformers

from .configs import config

def clean_text(text: str):
    text = html.unescape(text)
    text = text.replace("\n", "[NEWLINE]")
    text = text.replace("\r", "[NEWLINE]")
    text = ' '.join(text.split())
    return text

def get_section_encoding(heading: str, 
                         section: str, 
                         tokenizer: transformers.PreTrainedTokenizer)-> Tuple[List[int], List[int], Dict[str, int]]:
    """Returns a tuple having:
            1. encoding for the provided section & heading.
            2. component type labels for the encodings
            3. A dictionary mapping id of argumentative compoents
               to their end index in the encoding.
    """
    whole_content = heading+section
    whole_content = clean_text(whole_content)
    
    soup = BeautifulSoup("<xml>"+whole_content+"</xml>", "xml")
    
    tokenized_section, comp_type_labels = [tokenizer.bos_token_id], [config["arg_components"]["O"]]
    ids_to_ends = {}

    for component in soup.contents[0]:
        tokenized_component = tokenizer.encode(str(component.string))[1:-1]
        
        tokenized_section += tokenized_component
        
        if re.fullmatch(r"<background_claim id=\"T\d+\">.*</background_claim>", str(component)) is not None:
            ids_to_ends[component.attrs["id"]] = len(tokenized_component)
            comp_type_labels += [config["arg_components"]["B-BC"]] + [config["arg_components"]["I-BC"]]*(len(tokenized_component)-1)
        elif re.fullmatch(r"<own_claim id=\"T\d+\">.*</own_claim>", str(component)) is not None:
            ids_to_ends[component.attrs["id"]] = len(tokenized_component)
            comp_type_labels += [config["arg_components"]["B-OC"]] + [config["arg_components"]["I-OC"]]*(len(tokenized_component)-1)
        elif re.fullmatch(r"<data id=\"T\d+\">.*</data>", str(component)) is not None:
            ids_to_ends[component.attrs["id"]] = len(tokenized_component)
            comp_type_labels += [config["arg_components"]["B-D"]] + [config["arg_components"]["I-D"]]*(len(tokenized_component)-1)
        else:
            comp_type_labels += [config["arg_components"]["O"]]*len(tokenized_component)
    
    tokenized_section.append(tokenizer.eos_token_id)
    comp_type_labels.append(config["arg_components"]["O"])
    
    return tokenized_section, comp_type_labels, ids_to_ends

def get_in_section_rels(all_rel_types: List[Tuple[str, str, str]], 
                        ids_to_ends: Dict[str, int], 
                        max_len: int) -> List[Tuple[int, int, int]]:
    """
    Args:
        all_rel_types: A list of tuples of form (rel_type, arg1_id, arg2_id)
        ids_to_ends:   A dictionary mapping id of argumentative compoents
                       to their end index in some encoding. The keys(ids) are assumed 
                       to be in order of appearance of corres. components in the encoding.
        max_len:       The maximum length at which the encoding will be truncated.
    
    Returns:
        A list of tuples of form (rel_type_id, index of arg1 in the encoding, index of arg2 in the encoding)
        where:
            1. the rel type id is according to config["rel_type_to_id"] 
            2. and the index of an argumentative component is the position of that argumentative component 
               among all the argumentative componnents in the encoding. This index is 1-based.
    """
    in_section_rels = []
    
    #Component ids ordered according to their appearance in the tokenized section
    ordered_components = list(ids_to_ends.keys())
    
    for (rel_type, arg1_id, arg2_id) in all_rel_types:
        if ids_to_ends[arg1_id] <= max_len and ids_to_ends[arg2_id] <= max_len:
            in_section_rels.append((config["rel_type_to_id"][rel_type],
                                    ordered_components.index(arg1_id)+1, 
                                    ordered_components.index(arg2_id)+1))
    
    return in_section_rels
            
def tokenize_paper(heading_sections: List[Tuple[str, str]],
                   all_rel_types: List[Tuple[str, str, str]],
                   tokenizer: transformers.PreTrainedTokenizer,
                   separator_token_id: Optional[int]=None,
                   max_len: int = 4096) -> List[Tuple[List[int],List[int]]]:
    """
    Args:
        heading_sections:      A list of tuples of form (heading, section content) corres. to 
                               the heading and section of a single paper.
        
        all_rel_types:         A list of tuples of form (rel_type, arg1_id, arg2_id) corres. to 
                               all the relations in the paper whose heading and sections
                               are provided.

        tokenizer:             A HF tokenizer to tokenize and merge together sections.
                               Possibly with added special tokens like [NEWLINE].
        
        separator_token_id:    Id for token denoting beginning of new section. If
                               not provided, we use tokenizer.sep_token_id.

        max_len:               The maximum length for the encoding of a sub-part
                               of paper.
    Returns:
        List of tuples of form:
            (encodings of sub-parts of paper, component type labels of sub-parts, relation annotations). 
        
        Where:
            1. each sub-part consists of a set of contiguous sections, s.t., the encoding of the sub-part 
               has less than max_len tokens. 
            2. And relation annotations consist of tuples of form 
               (rel_type_id, index of arg1 in the encoding, index of arg2 in the encoding), where :
                    a.) the index indicates the position of the argumentative component among other arg. 
                        comps. in the encoding.
                    b.) These indices are 1-based. Only relations b/w components in the same sub-part are 
                        returned.
    """
    if separator_token_id is None:
        separator_token_id = tokenizer.sep_token_id

    sub_parts = []

    tokenized_sub_part, sub_part_labels = [tokenizer.bos_token_id], [config["arg_components"]["O"]]
    paper_ids_to_ends = {}

    for heading, section_content in heading_sections:
        tokenized_section, comp_type_labels, ids_to_ends = get_section_encoding(heading, 
                                                                                section_content,
                                                                                tokenizer)
        tokenized_section = tokenized_section[1:-1]
        comp_type_labels = comp_type_labels[1:-1]
        ids_to_ends = {k: v-1+len(tokenized_sub_part) for k, v in ids_to_ends.items()}

        if (len(tokenized_sub_part)!=1 and
            len(tokenized_sub_part)+len(tokenized_section)+2>=max_len):             #+1 for eos_token, +1 for separator_token_id
            
            tokenized_sub_part = tokenized_sub_part[:max_len-1]
            tokenized_sub_part.append(tokenizer.eos_token_id)
            sub_part_labels = sub_part_labels[:max_len-1]
            sub_part_labels.append(config["arg_components"]["O"])
        
            rel_anns = get_in_section_rels(all_rel_types, paper_ids_to_ends, max_len)
            sub_parts.append((tokenized_sub_part, sub_part_labels, rel_anns))
            
            tokenized_sub_part, sub_part_labels = [tokenizer.bos_token_id], [config["arg_components"]["O"]]

        paper_ids_to_ends.update(ids_to_ends)
        tokenized_sub_part += tokenized_section + [separator_token_id]
        sub_part_labels += comp_type_labels + [config["arg_components"]["O"]]
    
    if len(tokenized_sub_part)>1:
        tokenized_sub_part[:max_len-1].append(tokenizer.eos_token_id)
        sub_part_labels[:max_len-1].append(config["arg_components"]["O"])
        rel_anns = get_in_section_rels(all_rel_types, paper_ids_to_ends, max_len)
        sub_parts.append((tokenized_sub_part, sub_part_labels, rel_anns))
    
    return sub_parts
