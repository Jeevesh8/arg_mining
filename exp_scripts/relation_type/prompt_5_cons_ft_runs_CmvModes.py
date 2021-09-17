import warnings, random
from typing import Tuple, List, Union
warnings.filterwarnings('ignore')

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from transformers import LongformerTokenizer, LongformerModel
from sklearn.metrics import precision_recall_fscore_support as prf_metric

from arg_mining.datasets.cmv_modes import load_dataset, data_config

ac_dict = data_config["arg_components"]
rel_type_dict = {rel : i for i, rel in enumerate(data_config["adv_relations_map"].keys())}
rel_type_dict.pop("None")

user_tokens = ["[UNU]"]+[f"[USER{i}]" for i in range(data_config["max_users"])]
num_mask_pos = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class precision_recall_fscore():

    def __init__(self):
        self.preds = []
        self.refs = []

    def add_batch(self, predictions, references):
        self.preds += predictions
        self.refs += references

    def compute(self):
        f1_metrics = {"precision" : {}, "recall" : {},
                      "f1": {}, "support": {}} 
        precision, recall, f1, supp = prf_metric(self.refs, self.preds, 
                                                 labels=list(rel_type_dict.keys()))

        for i, k in enumerate(rel_type_dict.keys()):
            f1_metrics["precision"][k] = precision[i]
            f1_metrics["recall"][k] = recall[i]
            f1_metrics["f1"][k] = f1[i]
            f1_metrics["support"][k] = supp[i]

        for avg in ["micro", "macro", "weighted"]:
            precision, recall, f1, supp = prf_metric(self.refs, self.preds,
                                                     labels=list(rel_type_dict.keys()), average=avg)
            f1_metrics[avg+"_avg"] = {}
            f1_metrics[avg+"_avg"]["precision"] = precision 
            f1_metrics[avg+"_avg"]["recall"] = recall
            f1_metrics[avg+"_avg"]["f1"] = f1
            f1_metrics[avg+"_avg"]["support"] = supp

        self.preds = []
        self.refs = []

        return f1_metrics

metric = precision_recall_fscore()

def get_tok_model(tokenizer_version, model_version):
    tokenizer = LongformerTokenizer.from_pretrained(tokenizer_version)
    transformer_model = LongformerModel.from_pretrained(model_version).to(device)
    if tokenizer_version=='allenai/longformer-base-4096':
        tokenizer.add_tokens(data_config["special_tokens"])
    if model_version=='allenai/longformer-base-4096':
        transformer_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, transformer_model

"""#### Load in discourse markers"""

with open('./Discourse_Markers.txt') as f:
    discourse_markers = [dm.strip() for dm in f.readlines()]

"""#### Function to get train, test data (50/50 split currently)"""

def get_datasets(train_sz=100, test_sz=0):
    train_dataset, valid_dataset, test_dataset = load_dataset(tokenizer=tokenizer,
                                                              train_sz=train_sz,
                                                              test_sz=test_sz,
                                                              shuffle=True,
                                                              mask_tokens=discourse_markers)
    return train_dataset, valid_dataset, test_dataset

"""### Define linear layer for a relation type prediction"""

def get_rel_head():
    linear_layer = nn.Linear(num_mask_pos*transformer_model.config.hidden_size,
                             len(rel_type_dict)).to(device)

    return linear_layer

"""### Global Attention Mask Utility for Longformer"""

def get_global_attention_mask(tokenized_threads: np.ndarray) -> np.ndarray:
    """Returns an attention mask, with 1 where there are [USER{i}] tokens and 
    0 elsewhere.
    """
    mask = np.zeros_like(tokenized_threads)
    for user_token in user_tokens:
        user_token_id = tokenizer.encode(user_token)[1:-1]
        mask = np.where(tokenized_threads==user_token_id, 1, mask)
    return np.array(mask, dtype=bool)

def get_spans(comp_type_labels, length):
    
    def detect_span(start_idx: int, span_t: str):
        j = start_idx
        while comp_type_labels[j]==ac_dict[span_t] and j<length:
            j += 1
        end_idx = j
        return start_idx, end_idx
    
    i=0
    spans_lis = []
    while i<length:
        if comp_type_labels[i]==ac_dict["O"]:
            _start_idx, end_idx = detect_span(i, "O")

        elif comp_type_labels[i]==ac_dict["B-C"]:
            _start_idx, end_idx = detect_span(i+1, "I-C")
            spans_lis.append((i, end_idx))
            
        elif comp_type_labels[i]==ac_dict["B-P"]:
            _start_idx, end_idx = detect_span(i+1, "I-P")
            spans_lis.append((i, end_idx))
        
        elif (comp_type_labels[i]==ac_dict["I-C"] or
              comp_type_labels[i]==ac_dict["I-P"]):
            raise AssertionError("Span detection not working properly, \
                                  Or intermediate tokens without begin tokens in",
                                 comp_type_labels)
        else:
            raise ValueError("Unknown component type:", comp_type_labels[i], 
                             "Known types are:", ac_dict)
        
        i = end_idx
    
    return spans_lis

def get_user_token_positions(tokenized_thread):
    user_token_indices = [tokenizer.encode(user_token)[1] for user_token in user_tokens]
    user_token_pos = []
    for i, elem in enumerate(tokenized_thread):
        if elem in user_token_indices:
            user_token_pos.append(i)
    return user_token_pos

def generate_prompts(tokenized_thread, comp_type_labels, refers_to_and_type):

    length = np.sum(tokenized_thread!=tokenizer.pad_token_id)
    spans_lis = get_spans(comp_type_labels, length)
    print("First two spans:", spans_lis[0], spans_lis[1])
    user_token_pos = get_user_token_positions(tokenized_thread)
    for (link_from, link_to, rel_type) in refers_to_and_type:
        
        if link_to==0:
            continue
        
        from_start_idx, from_end_idx = spans_lis[link_from-1]
        to_start_idx, to_end_idx = spans_lis[link_to-1]
        
        print("Link from:", tokenizer.decode(tokenized_thread[from_start_idx:from_end_idx]), from_start_idx, from_end_idx)
        print("Link to:", tokenizer.decode(tokenized_thread[to_start_idx:to_end_idx]), to_start_idx, to_end_idx)
        
        from_user_token_id, to_user_token_id = 0, 0
        
        for elem in user_token_pos:
            if elem<=from_start_idx:
                from_user_token_id = tokenized_thread[elem]
            if elem<=to_start_idx:
                to_user_token_id = tokenized_thread[elem]
            if elem>from_start_idx and elem>to_start_idx:
                break
        
        from_seq_length = from_end_idx-from_start_idx
        to_seq_length = to_end_idx - to_start_idx
        
        prompt = np.copy(tokenized_thread)
        prompt[length] = to_user_token_id
        prompt[length+1] = tokenizer.encode(" said")[1]
        
        mask_pos = length+2+to_seq_length
        
        prompt[length+2:mask_pos] = tokenized_thread[to_start_idx:to_end_idx]
        prompt[mask_pos:mask_pos+num_mask_pos] = [tokenizer.mask_token_id]*num_mask_pos
        
        prompt[mask_pos+num_mask_pos] = from_user_token_id
        prompt[mask_pos+num_mask_pos+1] = tokenizer.encode(" said")[1]
        prompt[mask_pos+num_mask_pos+2:mask_pos+num_mask_pos+2+from_seq_length] = tokenized_thread[from_start_idx:from_end_idx]
        
        yield (prompt, rel_type)

def get_prompt_generator(dataset, batch_size, shuffle=True):
    prompt_dataset = []
    
    for (tokenized_threads, _, comp_type_labels, refers_to_and_type) in dataset:
        tokenized_threads = tokenized_threads[0]
        comp_type_labels = comp_type_labels[0]
        refers_to_and_type = refers_to_and_type[0]
        for (sample_tokenized_thread, 
            sample_comp_type_labels, 
            sample_refers_to_and_type) in zip(tokenized_threads, 
                                            comp_type_labels, 
                                            refers_to_and_type):
                
            prompt_dataset += [elem for elem in generate_prompts(sample_tokenized_thread, 
                                                                sample_comp_type_labels,
                                                                sample_refers_to_and_type)]
    if shuffle:
        random.shuffle(prompt_dataset)
    
    def prompt_dataset_gen():
        batch_of_prompts = []
        rel_type_labels = []
        for prompt, rel_type in prompt_dataset:
            batch_of_prompts.append(prompt)
            rel_type_labels.append(rel_type)
            if len(batch_of_prompts)==batch_size:
                yield batch_of_prompts, rel_type_labels
                batch_of_prompts, rel_type_labels = [], []
    
    return prompt_dataset_gen

"""### Loss and Prediction Function"""

cross_entropy_layer = nn.CrossEntropyLoss(weight=torch.tensor([0.508, 2.027, 2.402, 2.234, 2.667], 
                                                              device=device), reduction='sum')

def compute(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            preds: bool=False,) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Args:
        batch:  A tuple having prompting thread of shape [batch_size, seq_len],
                and relation type labels of shape [batch_size], and global attention mask
                for longformer of shape [batch_size, seq_len] with 1 for tokens which attend
                globally.
        
        preds:  If True, returns a List(of batch_size size) of relation predictions
                for correspoding prompts in the batch.
    Returns:
        Either the predicted relations(if preds is True), or the loss value 
        summed over all samples of the batch (if preds is False).
    """
    prompt_threads, rel_type_labels, global_attention_mask = batch
    
    pad_mask = torch.where(prompt_threads!=tokenizer.pad_token_id, 1, 0)
    
    hidden_state = transformer_model(input_ids=prompt_threads,
                                     attention_mask=pad_mask,
                                     global_attention_mask=global_attention_mask).last_hidden_state
    
    hidden_state = hidden_state[prompt_threads==tokenizer.mask_token_id]
    hidden_state = hidden_state.reshape(-1, num_mask_pos, hidden_state.shape[-1])
    hidden_state = torch.flatten(hidden_state, start_dim=1)
    rel_type_logits = linear_layer(hidden_state)

    if preds:
        return torch.max(rel_type_logits, dim=-1).indices
    
    ce_loss = cross_entropy_layer(rel_type_logits, rel_type_labels)

    return ce_loss

"""### Training And Evaluation Loops"""

def train(dataset):
    accumulate_over = 4
    
    optimizer.zero_grad()

    for i, (prompt_threads, rel_type_labels) in enumerate(
                                                    get_prompt_generator(dataset, 
                                                                         batch_size=data_config["batch_size"])):
        
        global_attention_mask = torch.tensor(get_global_attention_mask(prompt_threads),
                                             device=device, dtype=torch.int32)
        
        #Cast to PyTorch tensor
        prompt_threads = torch.tensor(prompt_threads, device=device)
        rel_type_labels = torch.tensor(rel_type_labels, device=device)
        global_attention_mask = torch.squeeze(global_attention_mask, dim=0)
        
        loss = compute((prompt_threads,
                        rel_type_labels,
                        global_attention_mask))/data_config["batch_size"]

        print("Loss:", loss)
        
        loss.backward()
        
        if i%accumulate_over==accumulate_over-1:
            optimizer.step()
            optimizer.zero_grad()
    
    optimizer.step()

def evaluate(dataset, metric):
    
    int_to_labels = {v:k for k, v in rel_type_dict.items()}

    with torch.no_grad():
        for prompt_threads, rel_type_labels in get_prompt_generator(dataset,
                                                                    batch_size=data_config["batch_size"]):
            print("Evaluating") 
            global_attention_mask = torch.tensor(get_global_attention_mask(tokenized_threads), 
                                                 device=device)
            
            #Cast to PyTorch tensor
            prompt_threads = torch.tensor(prompt_threads, device=device)
            rel_type_labels = torch.tensor(rel_type_labels, device=device)
            global_attention_mask = torch.squeeze(global_attention_mask, dim=0)
            
            preds = compute((prompt_threads,
                             rel_type_labels,
                             global_attention_mask),
                            preds=True)
            
            preds = [int_to_labels[pred.item()] for pred in preds]
            refs =  [int_to_labels[ref.item()] for ref in rel_type_labels]
            
            metric.add_batch(predictions=preds,
                             references=refs,)
        
    print("\t\t\t\t", metric.compute())

"""### Final Training"""

n_epochs = 30
n_runs = 5
for (tokenizer_version, model_version) in [('../home/arg_mining/4epoch_complete/tokenizer', '../home/arg_mining/4epoch_complete/model/'),
                                           #('arg_mining/smlm_pretrained_iter5_0/tokenizer/', 'arg_mining/smlm_pretrained_iter5_0/model/'),
                                           #('arg_mining/smlm_pretrained_iter6_0/tokenizer/', 'arg_mining/smlm_pretrained_iter6_0/model/'),
                                           #('arg_mining/smlm_pretrained_iter7_0/tokenizer/', 'arg_mining/smlm_pretrained_iter7_0/model/'),]:
                                           ('allenai/longformer-base-4096', 'allenai/longformer-base-4096')]:

    print("Tokenizer:", tokenizer_version, "Model:", model_version)
    
    for (train_sz, test_sz) in [(80,20),(50,50)]:
    
        print("\tTrain size:", train_sz, "Test size:", test_sz)
        
        for run in range(n_runs):
            print(f"\n\n\t\t-------------RUN {run+1}-----------")
            
            tokenizer, transformer_model = get_tok_model(tokenizer_version, model_version)

            linear_layer = get_rel_head()
            
            optimizer = optim.Adam(params = chain(transformer_model.parameters(),
                                      linear_layer.parameters(),),
                                   lr = 2e-5,)

            train_dataset, _, test_dataset = get_datasets(train_sz, test_sz)
            train_dataset = [elem for elem in train_dataset]
            test_dataset = [elem for elem in test_dataset]

            for epoch in range(n_epochs):
                print(f"\t\t\t------------EPOCH {epoch+1}---------------")
                evaluate(test_dataset, metric)
                train(train_dataset)
           
            del tokenizer, transformer_model, linear_layer
