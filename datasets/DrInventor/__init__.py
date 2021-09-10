import subprocess, shlex, os, random
from typing import Generator, List, Tuple, Optional

import transformers
from transformers import BertTokenizer

from .configs import config
from .tokenize import tokenize_paper
from .subparts import break_into_sections
from .basic_text_processing import refine, add_component_type_tags

def download_data():
    
    if not os.path.isdir("./compiled_corpus/"):
        subprocess.call(shlex.split("wget http://data.dws.informatik.uni-mannheim.de/sci-arg/compiled_corpus.zip"))
        subprocess.call(shlex.split("unzip compiled_corpus.zip"))
    
    return "./compiled_corpus/"

def data_generator(file_lis: List[Tuple[str, str]], 
                   tokenizer: transformers.PreTrainedTokenizer,
                   max_len: Optional[int] = 4096):
    
    for txt_file, ann_file in file_lis:
        with open(txt_file) as g:
            paper_str = g.read()
        with open(ann_file) as g:
            annotations = g.readlines()
        refined_annotations = refine(annotations)
        new_paper_str = add_component_type_tags(paper_str, refined_annotations)
        heading_sections = break_into_sections(new_paper_str, merge_subsecs=(max_len>1024))
        sub_parts = tokenize_paper(heading_sections, tokenizer, max_len=max_len)
        for sub_part in sub_parts:
            yield sub_part

def batched_data_gen(file_lis: List[Tuple[str, str]], 
                     tokenizer: transformers.PreTrainedTokenizer,
                     batch_sz: int,
                     pad_to_max: Optional[bool]=False,
                     max_len: Optional[int]=0,) -> Generator[Tuple[List[List[int]], List[List[int]]], None, None]:
    i=0
    batched_paper_parts, batched_labels = [],  []
    for tokenized_paper_part, comp_type_labels in data_generator(file_lis, tokenizer, max_len):
        batched_paper_parts.append(tokenized_paper_part)
        batched_labels.append(comp_type_labels)
        i += 1
        if i%batch_sz==0:
            if not pad_to_max:
                max_len = max([len(elem) for elem in batched_paper_parts])
            yield ([elem+[tokenizer.pad_token_id]*(max_len-len(elem)) for elem in batched_paper_parts],
                   [elem+[config["pad_for"]["arg_components"]]*(max_len-len(elem)) for elem in batched_labels])
            
            batched_paper_parts, batched_labels = [],  []


def load_dataset(dataset_dir: Optional[str] = None,
                 train_sz:    Optional[int] = 100,
                 valid_sz:    Optional[int] = 0,
                 test_sz:     Optional[int] = 0,
                 batch_sz:    Optional[int] = 4,
                 max_len:     Optional[int] = None,
                 tokenizer:   Optional[transformers.PreTrainedTokenizer] = None,
                 pad_to_max:  Optional[bool] = False,
                 shuffle:     Optional[bool] = False):

    assert train_sz+valid_sz+test_sz==100, "Train, Valid, Test size must sum to 100(%)"
    
    if dataset_dir is None:
        dataset_dir = download_data()
    
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', bos_token="[CLS]", eos_token="[SEP]")
        tokenizer.add_tokens(config["special_tokens"], special_tokens=True)
    
    if max_len is None:
        max_len = tokenizer.model_max_length
    
    files = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".txt") and os.path.isfile(os.path.join(dataset_dir, filename[:-4]+".ann")):
            files.append((os.path.join(dataset_dir, filename),
                          os.path.join(dataset_dir, filename[:-4]+".ann")))
    
    if shuffle:
        random.shuffle(files)
    
    num_train_files = (len(files)*train_sz)//100
    num_valid_files = (len(files)*valid_sz)//100
    num_test_files = len(files)-num_train_files-num_valid_files
    
    train_dataset = None if num_train_files==0 else batched_data_gen(files[:num_train_files], tokenizer, batch_sz, pad_to_max, max_len)
    valid_dataset = None if num_valid_files==0 else batched_data_gen(files[num_train_files:num_train_files+num_valid_files], tokenizer, batch_sz, pad_to_max, max_len)
    test_dataset  = None if num_test_files==0 else batched_data_gen(files[num_train_files+num_valid_files:], tokenizer, batch_sz, pad_to_max, max_len)

    return train_dataset, valid_dataset, test_dataset
