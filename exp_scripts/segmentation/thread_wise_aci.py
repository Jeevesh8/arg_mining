# Thread Level ACI on CMV Modes dataset. Change Models to load, splits to check in loops at bottom.
import warnings, argparse
warnings.filterwarnings('ignore')
from itertools import chain

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datasets import load_metric
from transformers import LongformerTokenizer, LongformerModel
from allennlp.modules.conditional_random_field import ConditionalRandomField as crf

from arg_mining.datasets.cmv_modes import load_dataset, data_config
from arg_mining.utils.misc import get_model_tok_version

metric = load_metric('seqeval')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_tok_model(tokenizer_version, model_version):
    tokenizer = LongformerTokenizer.from_pretrained(tokenizer_version)
    transformer_model = LongformerModel.from_pretrained(model_version).to(device)
    if tokenizer_version=='allenai/longformer-base-4096':
        tokenizer.add_tokens(data_config["special_tokens"])
    if model_version=='allenai/longformer-base-4096':
        transformer_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, transformer_model


def get_datasets(train_sz=100, test_sz=0):
    train_dataset, valid_dataset, test_dataset = load_dataset(tokenizer=tokenizer,
                                                              train_sz=train_sz,
                                                              test_sz=test_sz,
                                                              shuffle=True,
                                                              mask_tokens=discourse_markers)
    return train_dataset, valid_dataset, test_dataset


ac_dict = data_config["arg_components"]

allowed_transitions =([(ac_dict["B-C"], ac_dict["I-C"]), 
                       (ac_dict["B-P"], ac_dict["I-P"])] + 
                      [(ac_dict["I-C"], ac_dict[ct]) 
                        for ct in ["I-C", "B-C", "B-P", "O"]] +
                      [(ac_dict["I-P"], ac_dict[ct]) 
                        for ct in ["I-P", "B-C", "B-P", "O"]] +
                      [(ac_dict["O"], ac_dict[ct]) 
                        for ct in ["O", "B-C", "B-P"]])

cross_entropy_layer = nn.CrossEntropyLoss(weight=torch.log(torch.tensor([3.3102, 61.4809, 3.6832, 49.6827, 2.5639], 
                                                                        device=device)), reduction='none')

def get_crf_head():
    linear_layer = nn.Linear(transformer_model.config.hidden_size,
                             len(ac_dict)).to(device)

    crf_layer = crf(num_tags=len(ac_dict),
                    constraints=allowed_transitions,
                    include_start_end_transitions=False).to(device)

    return linear_layer, crf_layer


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

from typing import Tuple

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
    tokenized_threads, token_type_ids, comp_type_labels, global_attention_mask = batch
    
    pad_mask = torch.where(tokenized_threads!=tokenizer.pad_token_id, 1, 0)
    
    logits = linear_layer(transformer_model(input_ids=tokenized_threads,
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


"""### Training And Evaluation Loops"""

def train(dataset):
    accumulate_over = 4
    
    optimizer.zero_grad()

    for i, (tokenized_threads, masked_threads, comp_type_labels, _ ) in enumerate(dataset):
        global_attention_mask = torch.tensor(get_global_attention_mask(tokenized_threads),
                                             device=device, dtype=torch.int32)
        
        #Remove Device Axis and cast to PyTorch tensor
        tokenized_threads = torch.tensor(np.squeeze(tokenized_threads, axis=0), 
                                         device=device)
        masked_threads = torch.tensor(np.squeeze(masked_threads, axis=0), 
                                      device=device)
        comp_type_labels = torch.tensor(np.squeeze(comp_type_labels, axis=0), 
                                        device=device, dtype=torch.long)
        
        global_attention_mask = torch.squeeze(global_attention_mask, dim=0)
        
        loss = compute((tokenized_threads,
                        torch.where(masked_threads==tokenizer.mask_token_id, 1, 0), 
                        comp_type_labels, 
                        global_attention_mask))/data_config["batch_size"]

        print("Loss:", loss)
        
        loss.backward()
        
        if i%accumulate_over==accumulate_over-1:
            optimizer.step()
            optimizer.zero_grad()
    
    optimizer.step()

def evaluate(dataset, metric):
    
    int_to_labels = {v:k for k, v in ac_dict.items()}

    with torch.no_grad():
        for tokenized_threads, masked_threads, comp_type_labels, _ in dataset:
            print("Evaluating") 
            global_attention_mask = torch.tensor(get_global_attention_mask(tokenized_threads), 
                                                 device=device)
            
            #Remove Device Axis and cast to PyTorch tensor
            tokenized_threads = torch.tensor(np.squeeze(tokenized_threads, axis=0),
                                            device=device)
            masked_threads = torch.tensor(np.squeeze(masked_threads, axis=0),
                                         device=device)
            comp_type_labels = torch.tensor(np.squeeze(comp_type_labels, axis=0),
                                            device=device)
            global_attention_mask = torch.squeeze(global_attention_mask, dim=0)
            
            preds = compute((tokenized_threads,
                             torch.where(masked_threads==tokenizer.mask_token_id, 1, 0), 
                             comp_type_labels,
                             global_attention_mask),
                            preds=True)
            
            lengths = torch.sum(torch.where(tokenized_threads!=tokenizer.pad_token_id, 1, 0), 
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

def main(args):
    global discourse_markers
    with open(args.dm_file) as f:
        discourse_markers = [dm.strip() for dm in f.readlines()]
    
    tokenizer_version, model_version = get_model_tok_version(args)
    print("Tokenizer:", tokenizer_version, "Model:", model_version)

    global tokenizer, transformer_model, crf_layer, linear_layer, optimizer

    for (train_sz, test_sz) in [(80,20),(50,50)]:
        print("\tTrain size:", train_sz, "Test size:", test_sz)
        for run in range(args.n_runs):
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

            for epoch in range(args.n_epochs):
                print(f"\t\t\t------------EPOCH {epoch+1}---------------")
                train(train_dataset)
                evaluate(test_dataset, metric)
            
            del tokenizer, transformer_model, linear_layer, crf_layer

def get_parser():
    parser = argparse.ArgumentParser(description="Token level, thread wise Argument Component Span Detection and Classification on CMV Modes dataset with Longformer.")
    parser.add_argument("--pretrained", default="allenai/longformer-base-4096", help="Path to folder having pretrained weights, or the version of pretrained model to load.")
    parser.add_argument("--n_epochs", default=30, type=int, help="Number of epochs to train.")
    parser.add_argument("--n_runs", default=5, type=int, help="Number of runs to train.")
    parser.add_argument("--dm_file", default="./arg_mining/Discourse_Markers.txt", help="File having discourse markers.")
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)