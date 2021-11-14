# Relation Type Prediction

## Context-Less Fine-Tuning

To do context-less finetuning for RoBERTa, QR-BERT models run the following commands:
```
python3 contextless_RTP.py --model_type [bert|roberta] --pretrained [PRETRAINED_MODEL] \
                           --n_epochs <number of epochs> --n_runs <number of runs> \
                           --dm_file <Path to Discourse Markers file>
```
where ``[PRETRAINED_MODEL]`` is either ``roberta-base`` or path to QR-Finetuned Model of [AMPERSAND](https://github.com/tuhinjubcse/AMPERSAND-EMNLP2019) provided [here](https://drive.google.com/file/d/1wWs_0pb2N9dmXz6RjnW7TiJkV-b1m9Np/view).

## Mean Pooling Fine-Tuning

To do mean-pooling Fine-Tuning on CMV-Modes dataset, run:
```
python3 mean_pooling_RTP.py --pretrained [PRETRAINED_MODEL] --n_epochs <number of epochs> --n_runs <number of runs> \
                            --dm_file <Path to Discourse Markers file>
```
where ``[PRETRAINED_MODEL]`` is ``allenai/longformer-base-4096``(default) or any of the other LongFormer based pretrained models provided by us.

## Prompting RTP Fine-Tuning

To do prompting fine-tuning on CMV-Modes dataset, run:
```
python3 prompt_RTP.py --pretrained [PRETRAINED_MODEL] --n_epochs <number of epochs> --n_runs <number of runs> \
                      --dm_file <Path to Discourse Markers file>
```
where ``[PRETRAINED_MODEL]`` is ``allenai/longformer-base-4096``(default) or any of the other LongFormer based pretrained models provided by us.
