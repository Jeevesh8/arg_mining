import html, re
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional

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
                         tokenizer: transformers.PreTrainedTokenizer)-> Tuple[List[int], List[int]]:
    """Returns a tuple having encoding and labels for the provided section & heading."""
    whole_content = heading+section
    whole_content = clean_text(whole_content)
    
    soup = BeautifulSoup("<xml>"+whole_content+"</xml>", "xml")
    
    tokenized_section, comp_type_labels = [tokenizer.bos_token_id], [config["arg_components"]["other"]]
    
    for component in soup.contents[0]:
        tokenized_component = tokenizer.encode(str(component.string))[1:-1]
        
        tokenized_section += tokenized_component
        
        if re.fullmatch(r"<background_claim>.*</background_claim>", str(component)) is not None:
            comp_type_labels += [config["arg_components"]["B-BC"]] + [config["arg_components"]["I-BC"]]*(len(tokenized_component)-1)
        elif re.fullmatch(r"<own_claim>.*</own_claim>", str(component)) is not None:
            comp_type_labels += [config["arg_components"]["B-OC"]] + [config["arg_components"]["I-OC"]]*(len(tokenized_component)-1)
        elif re.fullmatch(r"<data>.*</data>", str(component)) is not None:
            comp_type_labels += [config["arg_components"]["B-D"]] + [config["arg_components"]["I-D"]]*(len(tokenized_component)-1)
        else:
            comp_type_labels += [config["arg_components"]["other"]]*len(tokenized_component)
    
    tokenized_section.append(tokenizer.eos_token_id)
    comp_type_labels.append(config["arg_components"]["other"])
    
    return tokenized_section, comp_type_labels


def tokenize_paper(heading_sections: List[Tuple[str, str]],
                   tokenizer: transformers.PreTrainedTokenizer,
                   separator_token_id: Optional[int]=None,
                   max_len: int = 4096) -> List[Tuple[List[int],List[int]]]:
    """
    Args:
        heading_sections:      A list of tuples of form (heading, section content)
        
        tokenizer:             A HF tokenizer to tokenize and merge together sections.
                               Possibly with added special tokens like [NEWLINE].
        
        separator_token_id:    Id for token denoting beginning of new section. If
                               not provided, we use tokenizer.sep_token_id.

        max_len:               The maximum length for the encoding of a sub-part
                               of paper.
    Returns:
        List of tuples of form :
            (encodings of sub-parts of paper, component type labels of sub-parts). 
        Where each sub-part consists of a set of contiguous sections, s.t., 
        the encoding of the sub-part has less than max_len tokens.
    """
    if separator_token_id is None:
        separator_token_id = tokenizer.sep_token_id

    sub_parts = []

    tokenized_sub_part, sub_part_labels = [tokenizer.bos_token_id], [config["arg_components"]["other"]]

    for heading, section_content in heading_sections:
        tokenized_section, comp_type_labels = get_section_encoding(heading, 
                                                                   section_content,
                                                                   tokenizer)
        tokenized_section = tokenized_section[1:-1]
        comp_type_labels = comp_type_labels[1:-1]
        
        if (len(tokenized_sub_part)!=1 and
            len(tokenized_sub_part)+len(tokenized_section)+2>max_len):             #+1 for eos_token, +1 for separator_token_id
            
            tokenized_sub_part = tokenized_sub_part[:max_len-1]
            tokenized_sub_part.append(tokenizer.eos_token_id)
            sub_part_labels = sub_part_labels[:max_len-1]
            sub_part_labels.append(config["arg_components"]["other"])
            
            sub_parts.append((tokenized_sub_part, sub_part_labels))
            
            tokenized_sub_part, sub_part_labels = [tokenizer.bos_token_id], [config["arg_components"]["other"]]

        tokenized_sub_part += tokenized_section + [separator_token_id]
        sub_part_labels += comp_type_labels + [config["arg_components"]["other"]]
    
    if len(tokenized_sub_part)>1:
        tokenized_sub_part[:max_len-1].append(tokenizer.eos_token_id)
        sub_part_labels[:max_len-1].append(config["arg_components"]["other"])
        sub_parts.append((tokenized_sub_part, sub_part_labels))
    
    return sub_parts
