# Argument Component Identification

## Thread Level
To do token level, thread wise Argument Component Span Detection and Classification on CMV Modes dataset with Longformer run:
```
python3 thread_wise_aci.py --pretrained [PRETRAINED_MODEL] --n_epochs <number of epochs>\
                           --n_runs <number of runs> --dm_file <Path to Discourse Markers>
```
where ``[PRETRAINED_MODEL]`` can be either path to folder having pretrained weights/tokenizer, or the version of pretrained model to load from HuggingFace Hub.

## Comment Level 
To do token level, comment wise Argument Component Span Detection and Classification on CMV Modes dataset, run:
```
python3 thread_wise_aci.py --model_type [bert|roberta|longformer] --pretrained [PRETRAINED_MODEL] --n_epochs <number of epochs> \
                           --n_runs <number of runs> --dm_file <Path to Discourse Markers>
```
where ``[PRETRAINED_MODEL]`` is same as before. 