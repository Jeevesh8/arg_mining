import re

import numpy as np
import torch
import torch.optim as optim
from convokit import Corpus, download
from transformers import LongformerTokenizer, LongformerForMaskedLM

from ..datasets.cmv_modes import data_config
from ..datasets.cmv_modes.utils import reencode_mask_tokens
from ..datasets.cmv_modes.component_generator import footnote_regex, quote_regex, url_regex


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_version = 'allenai/longformer-base-4096'

corpus = Corpus(filename=download("winning-args-corpus"))

tokenizer = LongformerTokenizer.from_pretrained(model_version)
transformer_model = LongformerForMaskedLM.from_pretrained(model_version).to(device)

optimizer = optim.Adam(params = transformer_model.parameters(), lr = 1e-6,)

with open('./Discourse_Markers.txt') as f:
    discourse_markers = sorted([dm.strip() for dm in f.readlines()], key=lambda marker: len(marker), reverse=True)

def get_global_attention_mask(tokenized_threads: torch.Tensor) -> torch.Tensor:
    """Returns an attention mask, with 1 where there are [USER{i}] tokens and 
    0 elsewhere.
    """
    mask = torch.zeros_like(tokenized_threads)
    for user_token in ["UNU"]+[f"[USER{i}]" for i in range(data_config["max_users"])]:
        user_token_id = tokenizer.encode(user_token)[1:-1]
        mask = torch.where(tokenized_threads==user_token_id, 1, mask)
    return torch.tensor(mask, dtype=bool, device=device)

def add_tags(text: str):
    text = re.sub(footnote_regex, "", text)
    text = re.sub(url_regex, "[URL]", text)
    text = re.sub(quote_regex, "[STARTQ]" + r"\1" + "[ENDQ] ", text)
    text = text.replace("\n", "[NEWLINE]")
    text = text.replace("\r", "[NEWLINE]")
    return text

def batch_generator(gen, batch_size=2, max_len=4096):
    batched_elem1, batched_elem2 = [], []
    for elem1, elem2 in gen:
        batched_elem1.append(elem1[:max_len])
        batched_elem2.append(elem2[:max_len])
        batched_elem1[-1] += [tokenizer.pad_token_id]*(max_len-len(batched_elem1[-1]))
        batched_elem2[-1] += [tokenizer.pad_token_id]*(max_len-len(batched_elem2[-1]))
        if len(batched_elem1)==batch_size:
            yield torch.tensor(batched_elem1, device=device), torch.tensor(batched_elem2, device=device)
            batched_elem1, batched_elem2 = [], []

def generate_thread_texts():
    for elem in corpus.iter_objs("conversation"):
        for path in elem.get_root_to_leaf_paths():
            thread_text = elem.meta["op-title"]
            for utterance in path:
                thread_text += utterance.text
            yield add_tags(thread_text)

dataset_len = 0
for elem in generate_thread_texts():
    dataset_len += 1

print("Number of threads in dataset:", dataset_len)

@batch_generator
def tokenized_ids_generator(start=0, end=100):
    for i, thread_text in enumerate(generate_thread_texts()):
        
        if (i/dataset_len)*100<start or (i/dataset_len)*100>end:
            continue
        
        encoding = tokenizer.encode(thread_text)
        label_encoding, masked_encoding = reencode_mask_tokens(encoding,
                                                               tokenizer,
                                                               discourse_markers)
        yield masked_encoding, label_encoding

def train():
    for masked_encoding, label_encoding in tokenized_ids_generator(0, 99):
        optimizer.zero_grad()
        
        loss = transformer_model(input_ids=masked_encoding,
                                 attention_mask=label_encoding!=tokenizer.pad_token_id,
                                 global_attention_mask=get_global_attention_mask(label_encoding),
                                 labels=label_encoding).loss
        
        loss.backward()
        optimizer.step()

def eval():
    with torch.no_grad():
        correct, total = 0, 0
        for masked_encoding, label_encoding in tokenized_ids_generator(99, 100):
            
            logits = transformer_model(input_ids=masked_encoding,
                                       attention_mask=label_encoding!=tokenizer.pad_token_id,
                                       global_attention_mask=get_global_attention_mask(label_encoding),
                                       labels=label_encoding).logits

            predictions = torch.max(logits, dim=-1)
            masked_positions = masked_encoding==tokenizer.mask_token_id
            correct_predictions = predictions==label_encoding
            correct += torch.sum(correct_predictions*masked_encoding)
            total += torch.sum(masked_positions)

    print("Test Accuracy:", correct/total)
    return correct/total

n_epochs = 10
save_dir = './smlm_pretrained/'

for i in range(n_epochs):
    print(f"--------------EPOCH {i+1}-------------")
    eval()
    train()
    transformer_model.save_pretrained(save_dir)