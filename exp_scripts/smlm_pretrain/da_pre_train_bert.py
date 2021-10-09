import re, os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.optim as optim
from convokit import Corpus, download
from transformers import BertTokenizer, BertForMaskedLM

from datasets.cmv_modes import data_config
from datasets.cmv_modes.utils import reencode_mask_tokens
from datasets.cmv_modes.component_generator import footnote_regex, quote_regex, url_regex

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_version = 'bert-base-cased'

corpus = Corpus(filename=download("winning-args-corpus"))

tokenizer = BertTokenizer.from_pretrained(model_version,
                                          bos_token = "[CLS]",
                                          eos_token = "[SEP]")

tokenizer.add_tokens(data_config["special_tokens"], special_tokens=True)

transformer_model = BertForMaskedLM.from_pretrained(model_version).to(device)
transformer_model.resize_token_embeddings(len(tokenizer))

optimizer = optim.Adam(params = transformer_model.parameters(), 
                       lr = 1e-6,)

with open('../Discourse_Markers.txt') as f:
    discourse_markers = sorted([dm.strip() for dm in f.readlines()], key=lambda marker: len(marker), reverse=True)

def add_tags(text: str):
    text = re.sub(footnote_regex, "", text)
    text = re.sub(url_regex, "[URL]", text)
    text = re.sub(quote_regex, "[STARTQ]" + r"\1" + "[ENDQ] ", text)
    text = text.replace("\n", "[NEWLINE]")
    text = text.replace("\r", "[NEWLINE]")
    return text

def batch_generator(gen, start, end, batch_size=1024, max_len=512):
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

def generate_utt_texts():
    for elem in corpus.iter_objs("conversation"):
        try:
            for utt in elem.iter_utterances():
                yield add_tags(utt.text)
        except ValueError:
            print("Encounterd erroneous tree! Skipping..")

sorted_utt_texts = sorted(list(generate_utt_texts()),
                             key=lambda item: len(tokenizer.encode(item)),
                             reverse=True)

dataset_len = 0
for elem in sorted_utt_texts:
    dataset_len += 1

print("Number of threads in dataset:", dataset_len)

def random_mask(encoding, tokenizer):
    encoding = np.array(encoding)
    masked_encoding = encoding.copy()
    rand_vals = np.random.default_rng().uniform(size=encoding.shape)
    
    special_toks = ([tokenizer.convert_tokens_to_ids(elem) for elem in tokenizer.special_tokens_map.values()]+
                    [elem for elem in tokenizer.get_added_vocab().values()])
    
    specail_toks_mask = np.zeros(encoding.shape)
    for token_id in special_toks:
        specail_toks_mask[encoding==token_id] = 1
    
    masked_encoding[np.logical_and(rand_vals<0.15, np.logical_not(specail_toks_mask))] = tokenizer.mask_token_id
    
    return encoding.tolist(), masked_encoding.tolist()

def tokenized_ids_generator(start=0, end=100):
    for i, utt_text in enumerate(sorted_utt_texts):
        
        if (i/dataset_len)*100<start or (i/dataset_len)*100>end:
            continue
        
        encoding = tokenizer.encode(utt_text)
        label_encoding, masked_encoding = random_mask(encoding,
                                                      tokenizer,)
        yield masked_encoding, label_encoding


n_epochs = 10
save_dir = './smlm_pretrained_iter'

def train(epoch_no, accumulate_over=3, save_ckpt_iters=80000):
    for i, (masked_encoding, label_encoding) in enumerate(batch_generator(tokenized_ids_generator, 0, 99)):
        
        loss = transformer_model(input_ids=masked_encoding,
                                 attention_mask=label_encoding!=tokenizer.pad_token_id,
                                 labels=label_encoding).loss

        print("Loss:", loss)
        loss.backward()
        
        if (i+1)%accumulate_over==0:
            optimizer.step()
            optimizer.zero_grad()
        
        if i%save_ckpt_iters==0:
            print("Saving model at iteration:", i)
            transformer_model.save_pretrained(os.path.join(save_dir+str(epoch_no)+'_'+str(i), 'model'))
            tokenizer.save_pretrained(os.path.join(save_dir+str(epoch_no)+'_'+str(i), 'tokenizer'))

def eval():
    with torch.no_grad():
        correct, total = 0, 0
        for masked_encoding, label_encoding in batch_generator(tokenized_ids_generator, 99, 100):
            
            logits = transformer_model(input_ids=masked_encoding,
                                       attention_mask=label_encoding!=tokenizer.pad_token_id,
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
    train(epoch_no=i+1)
