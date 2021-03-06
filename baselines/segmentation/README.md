# Create Environment to Run Baseline models
```
conda create --name arg_baselines python=3.6
conda activate arg_baselines
conda install bs4 lxml yaml pyyaml
pip install -r emnlp2017-bilstm-cnn-crf/requirements.txt
pip install seqeval
```

# Train and Evaluate Baseline

Clone the repo using ``git clone --recursive`` to make sure submodules are also cloned. Or clone the baseline repos in this directory using:
```
git clone http://github.com/UKPLab/naacl18-multitask_argument_mining
git clone https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf
```

## Multi-Task LSTM
```
bash multiTask.sh > out_file
```

## Multi-Data LSTM
```
bash multiData.sh > out_file
```
