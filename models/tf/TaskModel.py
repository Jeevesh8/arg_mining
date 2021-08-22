import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
import numpy as np

from .utils import get_transition_mat
from .configs import config


def compute_dsc_loss(y_pred, y_true, alpha=0.6):
    y_pred = K.reshape(K.softmax(y_pred), (-1, y_pred.shape[2]))
    y = K.expand_dims(K.flatten(y_true), axis=1)
    probs = tf.gather_nd(y_pred, y, batch_dims=1)
    pos = K.pow(1 - probs, alpha) * probs
    dsc_loss = 1 - (2 * pos + 1) / (pos + 2)
    return dsc_loss

class CRF(tf.keras.layers.Layer):
    def __init__(self, transition_matrix, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transitions = tf.Variable(transition_matrix)
    def call(self, inputs, mask=None, training=None):
        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)

        viterbi_sequence, _score = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )
        if training:
            return viterbi_sequence, inputs, sequence_lengths, self.transitions

        return viterbi_sequence, sequence_lengths

class TaskModel(tf.keras.models.Model):
    def __init__(self, encoder,
                 max_trans=0.1,
                 min_trans=-0.1,
                 is_padded=False,
                 use_gru=False,
                 alpha=0.5, lr=1e-5):
        super(TaskModel, self).__init__()
        self.encoder = encoder
        num_classes = len(config['arg_components']) #+1 Changed SD 21-06-2021
        self.ff = tf.keras.layers.Dense(num_classes)
        self.use_gru = use_gru
        self.alpha = alpha
        if use_gru:
            self.gru = tf.keras.layers.GRU(num_classes, return_sequences=True)
        self.crf_layer = CRF(get_transition_mat(min_val=min_trans, max_val=max_trans))

    def call(self, inputs, training=True):
        encoded_seq = self.encoder(inputs, training=training)['last_hidden_state']
        logits = self.gru(encoded_seq) if self.use_gru else self.ff(encoded_seq)
        crf_predictions = self.crf_layer(logits, mask=inputs['attention_mask'], training=training)

        if not training:
            return crf_predictions

        return tuple((*crf_predictions, logits))

    def compute_batch_sample_weight(self, labels, pad_mask, n_labels):
        counts = tf.reduce_sum(tf.cast(tf.equal(tf.expand_dims(tf.range(0, n_labels), -1), tf.reshape(labels, [-1])), dtype=tf.int32), axis=-1)
        counts = tf.cast(counts, dtype=tf.float32) + tf.keras.backend.epsilon()
        class_weights = tf.math.log(tf.reduce_sum(counts)/counts)
        non_pad = tf.cast(pad_mask, dtype=tf.float32)
        weighted_labels = tf.gather(class_weights, labels)
        return non_pad*weighted_labels

    def get_cross_entropy(self, logits, labels, pad_mask, n_labels):
        sample_weight = self.compute_batch_sample_weight(labels, pad_mask, n_labels=n_labels)
        cc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        n_samples = tf.reduce_sum(sample_weight)
        return tf.reduce_sum(cc_loss*sample_weight)/n_samples if n_samples!=0 else tf.convert_to_tensor(0.)

    def compute_loss(self, x, y):
        comp_type_labels = y
        _viterbi_sequence, potentials, sequence_length, chain_kernel, logits = self.call(x, training=True)
        crf_loss = -crf_log_likelihood(potentials, comp_type_labels, sequence_length, chain_kernel)[0]
        comp_type_cc_loss = self.get_cross_entropy(logits, comp_type_labels, x['attention_mask'], len(config['arg_components'])) #Changed SD 23.06.2021
        return tf.reduce_mean(crf_loss), comp_type_cc_loss # Changed SD 23.06.2021

    def infer_step(self, x):
        viterbi_seqs, seq_lens = self(x, training=False)
        return viterbi_seqs, seq_lens
    