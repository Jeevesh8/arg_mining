import re, os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.optim as optim
from convokit import Corpus, download
from transformers import LongformerTokenizer, LongformerForMaskedLM

from datasets.cmv_modes import data_config
from datasets.cmv_modes.utils import reencode_mask_tokens
from datasets.cmv_modes.component_generator import footnote_regex, quote_regex, url_regex

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_version = 'allenai/longformer-base-4096'

corpus = Corpus(filename=download("winning-args-corpus"))

tokenizer = LongformerTokenizer.from_pretrained(model_version)
tokenizer.add_tokens(data_config["special_tokens"])

transformer_model = LongformerForMaskedLM.from_pretrained(model_version).to(device)
transformer_model.resize_token_embeddings(len(tokenizer))

optimizer = optim.Adam(params = transformer_model.parameters(), lr = 1e-6,)

with open('../Discourse_Markers.txt') as f:
    discourse_markers = sorted([dm.strip() for dm in f.readlines()], key=lambda marker: len(marker), reverse=True)

def get_global_attention_mask(tokenized_threads: torch.Tensor) -> torch.Tensor:
    """Returns an attention mask, with 1 where there are [USER{i}] tokens and 
    0 elsewhere.
    """
    mask = torch.zeros_like(tokenized_threads)
    for user_token in ["UNU"]+[f"[USER{i}]" for i in range(data_config["max_users"])]:
        user_token_id = tokenizer.encode(user_token)[1]
        mask = torch.where(tokenized_threads==user_token_id, 1, mask)
    return torch.tensor(mask, dtype=bool, device=device)

def add_tags(text: str):
    text = re.sub(footnote_regex, "", text)
    text = re.sub(url_regex, "[URL]", text)
    text = re.sub(quote_regex, "[STARTQ]" + r"\1" + "[ENDQ] ", text)
    text = text.replace("\n", "[NEWLINE]")
    text = text.replace("\r", "[NEWLINE]")
    return text

def batch_generator(gen, start, end, batch_size=8194, max_len=4096):
    batched_elem1, batched_elem2, lengths = [], [], []
    for elem1, elem2 in gen(start, end):
        batched_elem1.append(elem1[:max_len])
        batched_elem2.append(elem2[:max_len])
        lengths.append(len(batched_elem1[-1]))
        if sum(lengths)>batch_size-max_len:
            for i in range(len(batched_elem1)):
                batched_elem1[i] += [tokenizer.pad_token_id]*(max(lengths)-len(batched_elem1[i]))
                batched_elem2[i] += [tokenizer.pad_token_id]*(max(lengths)-len(batched_elem2[i]))
 
            yield torch.tensor(batched_elem1, device=device), torch.tensor(batched_elem2, device=device)
            
            batched_elem1, batched_elem2, lengths = [], [], []

def generate_thread_texts():
    for elem in corpus.iter_objs("conversation"):
        try:
            for path in elem.get_root_to_leaf_paths():
                thread_text = elem.meta["op-title"]
                users_dict = {}
                for utterance in path:
                    spkr_id = utterance.get_speaker().id
                    if spkr_id=='[deleted]' or len(users_dict)>=data_config["max_users"]:
                        thread_text += "[UNU]"
                    else:
                        if spkr_id not in users_dict:
                            users_dict[spkr_id] = len(users_dict)
                        thread_text += "[USER"+str(users_dict[spkr_id])+"]"
                    thread_text += utterance.text
                yield add_tags(thread_text)
        except ValueError:
            print("Encountered erroneus tree! Skipping..")

sorted_thread_texts = sorted(list(generate_thread_texts()),
                             key=lambda item: len(tokenizer.encode(item)),
                             reverse=True)

dataset_len = 0
for elem in sorted_thread_texts:
    dataset_len += 1

print("Number of threads in dataset:", dataset_len)

def tokenized_ids_generator(start=0, end=100):
    for i, thread_text in enumerate(sorted_thread_texts):
        
        if (i/dataset_len)*100<start or (i/dataset_len)*100>end:
            continue
        
        encoding = tokenizer.encode(thread_text)
        label_encoding, masked_encoding = reencode_mask_tokens(encoding,
                                                               tokenizer,
                                                               discourse_markers)
        yield masked_encoding, label_encoding


n_epochs = 10
save_dir = './smlm_pretrained_iter'

def train(accumulate_over=3, save_ckpt_iters=20000):
    for i, (masked_encoding, label_encoding) in enumerate(batch_generator(tokenized_ids_generator, 0, 99)):
        
        loss = transformer_model(input_ids=masked_encoding,
                                 attention_mask=label_encoding!=tokenizer.pad_token_id,
                                 global_attention_mask=get_global_attention_mask(label_encoding),
                                 labels=label_encoding).loss

        print("Loss:", loss)
        loss.backward()
        
        if (i+1)%accumulate_over==0:
            optimizer.step()
            optimizer.zero_grad()
        
        if i%save_ckpt_iters==0:
            print("Saving model at iteration:", i)
            transformer_model.save_pretrained(os.path.join(save_dir+str(i), 'model'))
            tokenizer.save_pretrained(os.path.join(save_dir+str(i), 'tokenizer'))

def eval():
    with torch.no_grad():
        correct, total = 0, 0
        for masked_encoding, label_encoding in batch_generator(tokenized_ids_generator, 99, 100):
            
            logits = transformer_model(input_ids=masked_encoding,
                                       attention_mask=label_encoding!=tokenizer.pad_token_id,
                                       global_attention_mask=get_global_attention_mask(label_encoding),
                                       labels=label_encoding).logits

            predictions = torch.max(logits, dim=-1)[1]
            masked_positions = masked_encoding==tokenizer.mask_token_id
            correct_predictions = predictions==label_encoding
            correct += torch.sum(correct_predictions*masked_positions)
            total += torch.sum(masked_positions)

    print("Test Accuracy:", correct/total)
    return correct/total

for i in range(n_epochs):
    print(f"--------------EPOCH {i+1}-------------")
    eval()
    train()
