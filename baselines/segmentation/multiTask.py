from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

datasets = {
    'cmv_modes':
        {'columns': {1:'tokens', 2:'full_BIO'},
         'label': 'full_BIO',
         'evaluate': False,
         'commentSymbol': None},
    'cmv_modes1':
        {'columns': {1:'tokens', 3:'bt_BIO'},
         'label': 'bt_BIO',
         'evaluate': True,
         'commentSymbol': None},
    'cmv_modes2':
        {'columns': {1:'tokens', 4:'ds'},
         'label': 'ds',
         'evaluate': False,
         'commentSymbol': None},
}

embeddingsPath = 'komninos_english_embeddings.gz' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)

#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),
         'customClassifier': {'cmv_modes2': ['Softmax']},
          'earlyStopping': 40, 'charLSTMSize': 256, 'maxCharLength': 256,
          'miniBatchSize': 8}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=40)