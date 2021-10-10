import warnings, random, os
from itertools import chain
from typing import List, Tuple
warnings.filterwarnings('ignore')

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


from transformers import BertTokenizer, BertModel
from allennlp.modules.conditional_random_field import ConditionalRandomField as crf

from arg_mining.datasets.persuasive_essays import load_dataset, data_config
from arg_mining.utils.prf1 import precision_recall_fscore		   
        
metric = precision_recall_fscore(data_config["arg_components"])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_tok_model(tokenizer_version, model_version):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_version,
                                              bos_token = "[CLS]",
                                              eos_token = "[SEP]",)

    transformer_model = BertModel.from_pretrained(model_version).to(device)
    
    return tokenizer, transformer_model


def get_datasets(data_split="21k"):
    data_dir = os.path.join("./naacl18-multitask_argument_mining/dataSplits/", data_split, "essays/")
    train_dataset, _valid_dataset, test_dataset = load_dataset(pe_dir=data_dir, tokenizer=tokenizer)
    return train_dataset, test_dataset

### Define layers for a Linear-Chain-CRF

ac_dict = data_config["arg_components"]

allowed_transitions =([(ac_dict["B-C"], ac_dict["I-C"]), 
                       (ac_dict["B-P"], ac_dict["I-P"]),
                       (ac_dict["B-MC"], ac_dict["I-MC"])] + 
                      [(ac_dict["I-C"], ac_dict[ct]) 
                        for ct in ["I-C", "B-C", "B-P", "O", "B-MC"]] +
                      [(ac_dict["I-P"], ac_dict[ct]) 
                        for ct in ["I-P", "B-C", "B-P", "O", "B-MC"]] +
                      [(ac_dict["I-MC"], ac_dict[ct]) 
                        for ct in ["I-MC", "B-C", "B-P", "O"]] +
                      [(ac_dict["O"], ac_dict[ct])
                        for ct in ["O", "B-C", "B-P", "B-MC"]])
                    
def get_crf_head():
    linear_layer = nn.Linear(transformer_model.config.hidden_size,
                             len(ac_dict)).to(device)

    crf_layer = crf(num_tags=len(ac_dict),
                    constraints=allowed_transitions,
                    include_start_end_transitions=False).to(device)

    return linear_layer, crf_layer


cross_entropy_layer = nn.CrossEntropyLoss(weight=torch.tensor([1.1817, 4.6086, 1.9365, 3.6709, 0.8234, 5.31659, 2.66169],
                                                                        device=device), reduction='none')

"""### Loss and Prediction Function"""

def compute(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            preds: bool=False, cross_entropy: bool=True):
    """
    Args:
        batch:  A tuple having tokenized thread of shape [batch_size, seq_len],
                component type labels of shape [batch_size, seq_len], and a global
                attention mask for Longformer, of the same shape.
        
        preds:  If True, returns a List(of batch_size size) of Tuples of form 
                (tag_sequence, viterbi_score) where the tag_sequence is the 
                viterbi-decoded sequence, for the corresponding sample in the batch.
        
        cross_entropy:  This argument will only be used if preds=False, i.e., if 
                        loss is being calculated. If True, then cross entropy loss
                        will also be added to the output loss.
    
    Returns:
        Either the predicted sequences with their scores for each element in the batch
        (if preds is True), or the loss value summed over all elements of the batch
        (if preds is False).
    """
    tokenized_threads, _token_type_ids, comp_type_labels = batch
    
    pad_mask = torch.where(tokenized_threads!=tokenizer.pad_token_id, 1, 0)
    
    logits = linear_layer(transformer_model(input_ids=tokenized_threads,
                                            attention_mask=pad_mask,).last_hidden_state)
    
    if preds:
        return crf_layer.viterbi_tags(logits, pad_mask)
    
    log_likelihood = crf_layer(logits, comp_type_labels, pad_mask)
    
    if cross_entropy:
        logits = logits.reshape(-1, logits.shape[-1])
        
        pad_mask, comp_type_labels = pad_mask.reshape(-1), comp_type_labels.reshape(-1)
        
        ce_loss = torch.sum(pad_mask*cross_entropy_layer(logits, comp_type_labels))
        
        return ce_loss - log_likelihood

    return -log_likelihood


"""### Training And Evaluation Loops"""

def train(dataset):
    accumulate_over = 2

    optimizer.zero_grad()

    i=0
    for (tokenized_essays, comp_type_labels) in dataset:
        
        #Cast to PyTorch tensor
        tokenized_essays = torch.tensor(tokenized_essays, device=device)
        comp_type_labels = torch.tensor(comp_type_labels, device=device, dtype=torch.long)
        
        loss = compute((tokenized_essays,
                        torch.where(tokenized_essays==tokenizer.mask_token_id, 1, 0),
                        comp_type_labels,))/data_config["batch_size"]
        
        print("Loss: ", loss)
        loss.backward()
        
        if i%accumulate_over==accumulate_over-1:
            optimizer.step()
            optimizer.zero_grad()
        
        i += 1

    optimizer.step()

def evaluate(dataset, metric):
    
    int_to_labels = {v:k for k, v in ac_dict.items()}
    with torch.no_grad():
        for tokenized_essays, comp_type_labels in dataset:
        
            #Cast to PyTorch tensor
            tokenized_essays = torch.tensor(tokenized_essays, device=device)
            comp_type_labels = torch.tensor(comp_type_labels, device=device)
            
            preds = compute((tokenized_essays,
                            torch.where(tokenized_essays==tokenizer.mask_token_id, 1, 0), 
                            comp_type_labels,), preds=True)
            
            lengths = torch.sum(torch.where(tokenized_essays!=tokenizer.pad_token_id, 1, 0),
                                axis=-1)
            
            preds = [ [int_to_labels[pred] for pred in pred[0][:lengths[i]]]
                    for i, pred in enumerate(preds)
                    ]
            
            refs = [ [int_to_labels[ref] for ref in labels[:lengths[i]]]
                    for i, labels in enumerate(comp_type_labels.cpu().tolist())
                ]
            
            metric.add_batch(predictions=preds, 
                            references=refs,)
                            #tokenized_threads=tokenized_threads.cpu().tolist())
        
    print("\t\t\t\t", metric.compute())

"""### Final Training"""

n_epochs = 30
n_runs = 5

for (tokenizer_version, model_version) in [('bert-base-cased', 'bert-base-cased'),
                                           ('../arg_m/arg_mining/smlm_pretrained_iter5_0/tokenizer', '../arg_m/arg_mining/smlm_pretrained_iter5_0/model'),]:

    print("Tokenizer:", tokenizer_version, "Model:", model_version)

    for data_split in ["1k", "6k", "12k", "21k"]:

        print("\tData split:", data_split)

        tokenizer, transformer_model = get_tok_model(tokenizer_version, model_version)

        linear_layer, crf_layer = get_crf_head()

        optimizer = optim.Adam(params = chain(transformer_model.parameters(),
                                              linear_layer.parameters(), 
                                              crf_layer.parameters()),
                                lr = 2e-5,)

        train_dataset, test_dataset = get_datasets(data_split)
        train_dataset = [elem for elem in train_dataset]
        test_dataset = [elem for elem in test_dataset]

        for epoch in range(n_epochs):
            print(f"\t\t\t------------EPOCH {epoch+1}---------------")
            train(train_dataset)
            evaluate(test_dataset, metric)

        del tokenizer, transformer_model, linear_layer, crf_layer
