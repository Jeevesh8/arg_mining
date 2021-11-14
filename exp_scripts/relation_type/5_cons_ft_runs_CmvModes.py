#Mean Pooling RTP on CMV-Modes.
import warnings
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    linear_layer = nn.Linear(2*transformer_model.config.hidden_size,
                             len(rel_type_dict)).to(device)

    return linear_layer

cross_entropy_layer = nn.CrossEntropyLoss(weight=torch.tensor([0.508, 2.027, 2.402, 2.234, 2.667], 
                                                              device=device), reduction='sum')

"""### Global Attention Mask Utility for Longformer"""

def get_global_attention_mask(tokenized_threads: np.ndarray) -> np.ndarray:
    """Returns an attention mask, with 1 where there are [USER{i}] tokens and 
    0 elsewhere.
    """
    mask = np.zeros_like(tokenized_threads)
    for user_token in ["[UNU]"]+[f"[USER{i}]" for i in range(data_config["max_users"])]:
        user_token_id = tokenizer.encode(user_token)[1:-1]
        mask = np.where(tokenized_threads==user_token_id, 1, mask)
    return np.array(mask, dtype=bool)


"""### Loss and Prediction Function"""
def get_comp_wise_means(logits: torch.Tensor, 
                        comp_type_label: torch.Tensor,
                        tokenized_thread: torch.Tensor) -> torch.Tensor:
    """Averages logits, component wise.
    Args:
        logits:           logits of shape [seq_len, hidden_dim]
        
        comp_type_label:  argumentative component type of each token whose
                          logits are provided. Shape: [seq_len]
        
        tokenized_thread: the tokenized thread whose logits are provided.
                          Shape: [seq_len]
    Returns:
        An array of size [n_comps, hidden_dim] where n_comps is the number of 
        components in the comp_type_label. 
    """
    comp_means = []
    length = torch.sum(tokenized_thread!=tokenizer.pad_token_id)
    
    def detect_span(start_idx: int, span_t: str):
        j = start_idx
        while comp_type_label[j]==ac_dict[span_t] and j<length:
            j += 1
        end_idx = j
        return start_idx, end_idx
    
    i=0
    while i<length:
        if comp_type_label[i]==ac_dict["O"]:
            start_idx, end_idx = detect_span(i, "O")
        
        elif comp_type_label[i]==ac_dict["B-C"]:
            start_idx, end_idx = detect_span(i+1, "I-C")
            comp_means.append(torch.mean(logits[start_idx-1:end_idx], dim=0))
        
        elif comp_type_label[i]==ac_dict["B-P"]:
            start_idx, end_idx = detect_span(i+1, "I-P")
            comp_means.append(torch.mean(logits[start_idx-1:end_idx], dim=0))
        
        elif (comp_type_label[i]==ac_dict["I-C"] or 
              comp_type_label[i]==ac_dict["I-P"]):
            raise AssertionError("Span detection not working properly, \
                                  Or intermediate tokens without begin tokens in",
                                 comp_type_label)
        else:
            raise ValueError("Unknown component type:", comp_type_label[i], 
                             "Known types are:", ac_dict)
        
        i = end_idx
    
    return torch.stack(comp_means)

def relation_type_pred(comp_encodings: torch.Tensor, 
                       refers_to_and_type: np.ndarray) -> torch.Tensor:
    """
    Args:
        comp_encodings:     The encodings for various components in a 
                            tokenized_thread.
        refers_to_and_type: An array having entries of the form [i,j,k], where,
                            the entry denotes link from component i to component
                            j of type k. Component nos. are 1-indexed. Link to 0
                            indicates that component isn't linked to any other one.
    Returns:
        logits corresponding to relation types for relations(excluding the 
        relations to 0) in refers_to_and_type. The logits occur in the same order
        as the relations' orders in refers_to_and_type.
    """
    logits = []

    for (from_comp, to_comp, _rel_type) in refers_to_and_type:
        if to_comp==0:
            continue
            
        logits.append(linear_layer(torch.cat([comp_encodings[from_comp-1], 
                                              comp_encodings[to_comp-1]])))
        
    return torch.stack(logits)

def get_rel_types(refers_to_and_type: np.ndarray) -> torch.Tensor:
    """
    Args:
        refers_to_and_type:   Same as in relation_type_pred
    Returns:
        A pytorch tensor of type long having labels of various existant 
        relations.
    """
    label_rel_types = []
    for (_from_comp, to_comp, rel_type) in refers_to_and_type:
        if to_comp==0:
            continue
        label_rel_types.append(rel_type)
    return torch.tensor(label_rel_types, dtype=torch.long, device=device)

def compute(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            preds: bool=False,) -> Union[List[torch.Tensor], torch.Tensor]:
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
        Either the predicted relations(if preds is True), or the loss value 
        summed over all samples of the batch (if preds is False).
    """
    tokenized_threads, comp_type_labels, refers_to_and_type, global_attention_mask = batch
    
    pad_mask = torch.where(tokenized_threads!=tokenizer.pad_token_id, 1, 0)
    
    logits = transformer_model(input_ids=tokenized_threads,
                               attention_mask=pad_mask,
                               global_attention_mask=global_attention_mask).last_hidden_state
    
    comp_wise_logits = [get_comp_wise_means(sample_logits, sample_comp_type_labels, sample_tokenized_thread)
                            for (sample_logits, sample_comp_type_labels, sample_tokenized_thread)
                                 in zip(logits, comp_type_labels, tokenized_threads)]
    
    rel_type_logits =  [relation_type_pred(elem1, elem2) 
                        for elem1, elem2 in zip(comp_wise_logits, 
                                                refers_to_and_type)]
    
    
    if preds:
        return [torch.max(elem, dim=-1).indices for elem in rel_type_logits]
    
    rel_labels = [get_rel_types(elem) for elem in refers_to_and_type]

    ce_loss = sum([cross_entropy_layer(elem1, elem2)
                   for elem1, elem2 in zip(rel_type_logits, rel_labels)])
    
    return ce_loss

"""### Training And Evaluation Loops"""

def train(dataset):
    accumulate_over = 4
    
    optimizer.zero_grad()

    for i, (tokenized_threads, _, comp_type_labels, refers_to_and_type) in enumerate(dataset):
        global_attention_mask = torch.tensor(get_global_attention_mask(tokenized_threads),
                                             device=device, dtype=torch.int32)
        
        #Remove Device Axis and cast to PyTorch tensor
        tokenized_threads = torch.tensor(np.squeeze(tokenized_threads, axis=0), 
                                         device=device)
        comp_type_labels = torch.tensor(np.squeeze(comp_type_labels, axis=0), 
                                         device=device)
        refers_to_and_type = np.squeeze(refers_to_and_type, axis=0)

        global_attention_mask = torch.squeeze(global_attention_mask, dim=0)
        
        loss = compute((tokenized_threads,
                        comp_type_labels,
                        refers_to_and_type,
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
        for tokenized_threads, _, comp_type_labels, refers_to_and_type in dataset:
            print("Evaluating") 
            global_attention_mask = torch.tensor(get_global_attention_mask(tokenized_threads), 
                                                 device=device)
            
            #Remove Device Axis and cast to PyTorch tensor
            tokenized_threads = torch.tensor(np.squeeze(tokenized_threads, axis=0),
                                            device=device)
            comp_type_labels = torch.tensor(np.squeeze(comp_type_labels, axis=0),
                                            device=device)
            refers_to_and_type = np.squeeze(refers_to_and_type)

            global_attention_mask = torch.squeeze(global_attention_mask, dim=0)
            
            preds = compute((tokenized_threads,
                             comp_type_labels,
                             refers_to_and_type,
                             global_attention_mask),
                            preds=True)
            
            preds = [[int_to_labels[pred.item()] for pred in seq] for seq in preds]
            refs =  [[int_to_labels[ref.item()] for ref in get_rel_types(elem)] 
                     for elem in refers_to_and_type]
            
            for elem1, elem2 in zip(preds, refs):
                metric.add_batch(predictions=elem1,
                                 references=elem2,)
        
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
