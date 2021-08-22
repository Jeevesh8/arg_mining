from typing import List, Tuple

from .configs import config, tokenizer

def comp_type_from_tag(comp_tag):
    """Returns the type of component(claim/premise) from the tag 
       provided in persuasive essays dataset."""
    if comp_tag=='O':
        return 'O' 
    if 'Premise' in comp_tag:
        return 'P'
    return 'C'

def get_comp_wise_essays(data_file: str) -> List[List[Tuple[str, str]]]:
    """
    Args:
        data_file:  The file having essays("<word>\t<tag>" in each line).
    Returns:
        comp_wise_essays:  A list of all essays in data_file. Each essay is a list 
                           of components. Each component is a pair of (text, type(O/C/P)).
    """

    word_wise_essays = [[]]

    with open(data_file) as f:
        for line in f.readlines():
            if line.strip()!='':
                word_wise_essays[-1].append(line.split())
            else:
                word_wise_essays.append([])
        
    while word_wise_essays[-1]==[]:
        word_wise_essays = word_wise_essays[:-1]
    
    comp_wise_essays = []
    
    for essay in word_wise_essays:
        comp_wise_essay = []

        prev_comp_type = 'first'
        comp = ''
        for word, comp_type in essay:
            if prev_comp_type == 'first':
                comp += word+' '
            
            elif comp_type == 'O':
                if prev_comp_type=='O':
                    comp += word+' '
                else:
                    comp_wise_essay.append((comp.strip().lower(), comp_type_from_tag(prev_comp_type)))
                    comp = word + ' '
            
            elif comp_type.startswith('B'):
                comp_wise_essay.append((comp.strip().lower(), comp_type_from_tag(prev_comp_type)))
                comp = word + ' '
            
            elif comp_type.startswith('I'):
                comp += word + ' '
            
            else:
                raise ValueError("Unexpected component type: ", comp_type)
            
            prev_comp_type = comp_type
        
        comp_wise_essays.append(comp_wise_essay)
    
    return comp_wise_essays

def get_tokenized_essay(essay: List[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
    """Returns a tokenized essay and the corresponding target tags.
    Args:
        essay:  A list of components of an essay. Each component is tuple consisting of 
                the text of component and the type of component(O/C/P)
    Returns:
        tokenized_essay:    A list of ints corresponding to tokenization of each word of essay.
        comp_type_tags:     Int corres. to the tag ["other", "B-C", "I-C", "B-P", "I-P"] of each token in 
                            tokenized_essay.
    """
    
    tokenized_essay = [tokenizer.bos_token_id]
    comp_type_tags = [config["arg_components"]["other"]]

    for comp_text, comp_type in essay:
        tokenized_comp = tokenizer.encode(comp_text)[1:-1]
        
        if len(tokenized_comp)+len(tokenized_essay)>config["max_len"]-1:
            break
        
        tokenized_essay += tokenized_comp

        if comp_type=='O':
            comp_type_tags += [config["arg_components"]["other"]]*len(tokenized_comp)
        else:
            comp_type_tags += [config["arg_components"]["B-"+comp_type]] + [config["arg_components"]["I-"+comp_type]]*(len(tokenized_comp)-1)
    
    tokenized_essay.append(tokenizer.eos_token_id)
    comp_type_tags.append(config["arg_components"]["other"])

    return tokenized_essay, comp_type_tags


