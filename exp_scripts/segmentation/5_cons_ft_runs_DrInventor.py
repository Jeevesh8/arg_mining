# ACI on Dr. Inventor with Longformer. Change the models to load and split sizes in the loops at bottom. 
import warnings
warnings.filterwarnings('ignore')

from datasets import load_metric
metric = load_metric('seqeval')

"""### Define & Load Tokenizer, Model, Dataset"""
import numpy as np

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""#### Or load them from pretrained files..."""

from transformers import LongformerTokenizer, LongformerModel

from arg_mining.datasets.DrInventor import load_dataset
from arg_mining.datasets.DrInventor import config as data_config

def get_tok_model(tokenizer_version, model_version):
    tokenizer = LongformerTokenizer.from_pretrained(tokenizer_version)
    transformer_model = LongformerModel.from_pretrained(model_version).to(device)
    if tokenizer_version=='allenai/longformer-base-4096':
        tokenizer.add_tokens(data_config["special_tokens"])
    if model_version=='allenai/longformer-base-4096':
        transformer_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, transformer_model

import torch.nn as nn


"""#### Function to get train, test data (50/50 split currently)"""
batch_size = 2

def get_datasets(train_sz=100, test_sz=0):
    train_dataset, valid_dataset, test_dataset = load_dataset(tokenizer=tokenizer,
                                                              train_sz=train_sz,
                                                              test_sz=test_sz,
                                                              shuffle=True,
                                                              batch_sz=batch_size,)
    return train_dataset, valid_dataset, test_dataset

"""### Define layers for a Linear-Chain-CRF"""

from allennlp.modules.conditional_random_field import ConditionalRandomField as crf

ac_dict = data_config["arg_components"]

allowed_transitions =([(ac_dict["B-BC"], ac_dict["I-BC"]), 
                       (ac_dict["B-OC"], ac_dict["I-OC"]),
                       (ac_dict["B-D"], ac_dict["I-D"]),] + 
                      
                      [(ac_dict["I-BC"], ac_dict[ct]) 
                        for ct in ["I-BC", "B-BC", "B-OC","B-D", "O"]] +
                      [(ac_dict["I-OC"], ac_dict[ct]) 
                        for ct in ["I-OC", "B-BC", "B-OC", "B-D", "O"]] +
                      [(ac_dict["I-D"], ac_dict[ct]) 
                        for ct in ["I-D", "B-BC", "B-OC", "B-D", "O"]] +

                      [(ac_dict["O"], ac_dict[ct]) 
                        for ct in ["O", "B-BC", "B-OC", "B-D"]])

def get_crf_head():
    linear_layer = nn.Linear(transformer_model.config.hidden_size,
                             len(ac_dict)).to(device)

    crf_layer = crf(num_tags=len(ac_dict),
                    constraints=allowed_transitions,
                    include_start_end_transitions=False).to(device)

    return linear_layer, crf_layer

cross_entropy_layer = nn.CrossEntropyLoss(weight=torch.log(torch.tensor([0.668, 4.531, 2.020, 4.028, 1.388, 4.316, 2.754],
                                                                        device=device)), reduction='none')

"""### Loss and Prediction Function"""

from typing import Tuple

def compute(batch: Tuple[torch.Tensor, torch.Tensor],
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
    tokenized_sub_parts, comp_type_labels = batch
    
    pad_mask = torch.where(tokenized_sub_parts!=tokenizer.pad_token_id, 1, 0)

    global_attention_mask = torch.where(torch.logical_or(tokenized_sub_parts==tokenizer.sep_token_id,
                                                         tokenized_sub_parts==tokenizer.bos_token_id), 1, 0)

    logits = linear_layer(transformer_model(input_ids=tokenized_sub_parts,
                                            attention_mask=pad_mask,
                                            global_attention_mask=global_attention_mask).last_hidden_state)
    
    if preds:
        return crf_layer.viterbi_tags(logits, pad_mask)
    
    log_likelihood = crf_layer(logits, comp_type_labels, pad_mask)
    
    if cross_entropy:
        logits = logits.reshape(-1, logits.shape[-1])
        
        pad_mask, comp_type_labels = pad_mask.reshape(-1), comp_type_labels.reshape(-1)
        
        ce_loss = torch.sum(pad_mask*cross_entropy_layer(logits, comp_type_labels))
        
        return ce_loss - log_likelihood

    return -log_likelihood

"""### Define optimizer"""

from itertools import chain

import torch.optim as optim

"""### Training And Evaluation Loops"""

def train(dataset):
    accumulate_over = 4
    
    optimizer.zero_grad()

    for i, (tokenized_sub_parts, comp_type_labels, _) in enumerate(dataset):
        
        #Cast to PyTorch tensor
        tokenized_sub_parts = torch.tensor(tokenized_sub_parts, device=device)
        comp_type_labels = torch.tensor(comp_type_labels, device=device, dtype=torch.long)
        
        loss = compute((tokenized_sub_parts,
                        comp_type_labels,))/batch_size

        print("Loss:", loss)
        
        loss.backward()
        
        if i%accumulate_over==accumulate_over-1:
            optimizer.step()
            optimizer.zero_grad()
    
    optimizer.step()

def evaluate(dataset, metric):
    
    int_to_labels = {v:k for k, v in ac_dict.items()}

    with torch.no_grad():
        for tokenized_sub_parts, comp_type_labels, _ in dataset:
            print("Evaluating") 
           
            #Cast to PyTorch tensor
            tokenized_sub_parts = torch.tensor(tokenized_sub_parts, device=device)
            comp_type_labels = torch.tensor(comp_type_labels, device=device, dtype=torch.long)
            
            preds = compute((tokenized_sub_parts,
                             comp_type_labels,),
                            preds=True)
            
            lengths = torch.sum(torch.where(tokenized_sub_parts!=tokenizer.pad_token_id, 1, 0), 
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

n_epochs = 40
n_runs = 1
for (tokenizer_version, model_version) in [('arg_mining/4epoch_complete/tokenizer/', 'arg_mining/4epoch_complete/model/'),
                                           ('allenai/longformer-base-4096', 'allenai/longformer-base-4096')]:

    print("Tokenizer:", tokenizer_version, "Model:", model_version)
    
    for (train_sz, test_sz) in [(80,20),(50,50)]:
    
        print("\tTrain size:", train_sz, "Test size:", test_sz)
        
        for run in range(n_runs):
            print(f"\n\n\t\t-------------RUN {run+1}-----------")
            
            tokenizer, transformer_model = get_tok_model(tokenizer_version, model_version)
            
            linear_layer, crf_layer = get_crf_head()
            
            optimizer = optim.Adam(params = chain(transformer_model.parameters(),
                                      linear_layer.parameters(),
                                      crf_layer.parameters()),
                                   lr = 2e-5,)

            train_dataset, _, test_dataset = get_datasets(train_sz, test_sz)
            train_dataset = [elem for elem in train_dataset]
            test_dataset = [elem for elem in test_dataset]
            print("Train dataset size:", len(train_dataset))
            for epoch in range(n_epochs):
                print(f"\t\t\t------------EPOCH {epoch+1}---------------")
                train(train_dataset)
                evaluate(test_dataset, metric)
            
            del tokenizer, transformer_model, linear_layer, crf_layer
